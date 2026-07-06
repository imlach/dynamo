/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package operatorenv

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	goruntime "runtime"
	"sync"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/scale"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
)

type Options struct {
	Admission  bool
	Conversion bool

	OperatorVersion string
	Config          *configv1alpha1.OperatorConfiguration
	RuntimeConfig   *commoncontroller.RuntimeConfig

	CRDDirectoryPaths     []string
	BinaryAssetsDirectory string
	EventuallyTimeout     time.Duration
}

type Env struct {
	opts Options

	mu        sync.Mutex
	runM      bool
	shared    *runtimeEnv
	sharedErr error
	once      sync.Once
}

func New(opts Options) *Env {
	return &Env{opts: normalizeOptions(opts)}
}

func (e *Env) RunM(m *testing.M) int {
	e.mu.Lock()
	e.runM = true
	e.mu.Unlock()

	code := m.Run()
	if err := e.stopShared(); err != nil {
		fmt.Fprintf(os.Stderr, "operatorenv: stop shared env: %v\n", err)
		if code == 0 {
			code = 1
		}
	}
	return code
}

func (e *Env) RunT(tb testing.TB) *TestEnv {
	tb.Helper()
	rt, err := startRuntime(e.opts)
	if err != nil {
		tb.Fatalf("start isolated operatorenv: %v", err)
	}
	tb.Cleanup(func() {
		if err := rt.stop(); err != nil {
			tb.Errorf("stop isolated operatorenv: %v", err)
		}
	})
	return newTestEnv(tb, rt, e.opts)
}

func (e *Env) ForTest(tb testing.TB) *TestEnv {
	tb.Helper()
	e.mu.Lock()
	runM := e.runM
	e.mu.Unlock()
	if !runM {
		tb.Fatal("operatorenv.ForTest requires RunM from TestMain; use RunT for an isolated per-test env")
	}
	e.once.Do(func() {
		e.shared, e.sharedErr = startRuntime(e.opts)
	})
	if e.sharedErr != nil {
		tb.Fatalf("start shared operatorenv: %v", e.sharedErr)
	}
	return newTestEnv(tb, e.shared, e.opts)
}

func (e *Env) stopShared() error {
	e.mu.Lock()
	shared := e.shared
	e.mu.Unlock()
	if shared == nil {
		return nil
	}
	return shared.stop()
}

type runtimeEnv struct {
	opts          Options
	env           *envtest.Environment
	config        *rest.Config
	scheme        *k8sruntime.Scheme
	client        client.Client
	operatorCfg   *configv1alpha1.OperatorConfiguration
	runtimeConfig *commoncontroller.RuntimeConfig
	cancel        context.CancelFunc
	done          chan error
}

func startRuntime(opts Options) (*runtimeEnv, error) {
	scheme := newScheme()
	operatorCfg := defaultOperatorConfig(opts.Config)
	runtimeConfig := opts.RuntimeConfig
	if runtimeConfig == nil {
		runtimeConfig = &commoncontroller.RuntimeConfig{}
	}
	testEnv := &envtest.Environment{
		Scheme:                scheme,
		CRDDirectoryPaths:     crdDirectoryPaths(opts),
		ErrorIfCRDPathMissing: false,
		BinaryAssetsDirectory: binaryAssetsDirectory(opts),
		WebhookInstallOptions: webhookInstallOptions(opts),
	}
	cfg, err := testEnv.Start()
	if err != nil {
		return nil, err
	}
	k8sClient, err := client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		_ = testEnv.Stop()
		return nil, err
	}
	rt := &runtimeEnv{
		opts:          opts,
		env:           testEnv,
		config:        cfg,
		scheme:        scheme,
		client:        k8sClient,
		operatorCfg:   operatorCfg,
		runtimeConfig: runtimeConfig,
	}
	if opts.Admission || opts.Conversion {
		if err := rt.startWebhookManager(); err != nil {
			_ = testEnv.Stop()
			return nil, err
		}
	}
	return rt, nil
}

func (e *runtimeEnv) startWebhookManager() error {
	ctx, cancel := context.WithCancel(context.Background())
	server := webhook.NewServer(webhook.Options{
		Host:    e.env.WebhookInstallOptions.LocalServingHost,
		Port:    e.env.WebhookInstallOptions.LocalServingPort,
		CertDir: e.env.WebhookInstallOptions.LocalServingCertDir,
	})
	mgr, err := ctrl.NewManager(e.config, ctrl.Options{
		Scheme:        e.scheme,
		Metrics:       metricsserver.Options{BindAddress: "0"},
		WebhookServer: server,
	})
	if err != nil {
		cancel()
		return err
	}
	if err := webhooksetup.SetupAll(mgr, webhooksetup.Options{
		Config:          e.operatorCfg,
		RuntimeConfig:   e.runtimeConfig,
		OperatorVersion: e.opts.OperatorVersion,
	}); err != nil {
		cancel()
		return err
	}
	done := make(chan error, 1)
	go func() {
		done <- mgr.Start(ctx)
	}()
	if err := waitForWebhookServer(ctx, mgr.GetWebhookServer()); err != nil {
		cancel()
		<-done
		return err
	}
	e.cancel = cancel
	e.done = done
	return nil
}

func (e *runtimeEnv) stop() error {
	var errs []error
	if e.cancel != nil {
		e.cancel()
		if err := <-e.done; err != nil && !errors.Is(err, context.Canceled) {
			errs = append(errs, err)
		}
	}
	if e.env != nil {
		if err := e.env.Stop(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

type TestEnv struct {
	tb        testing.TB
	rt        *runtimeEnv
	namespace string
	opts      Options
}

func newTestEnv(tb testing.TB, rt *runtimeEnv, opts Options) *TestEnv {
	tb.Helper()
	name := "operatorenv-" + rand.String(8)
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if err := rt.client.Create(context.Background(), ns); err != nil {
		tb.Fatalf("create test namespace %q: %v", name, err)
	}
	tb.Cleanup(func() {
		if err := rt.client.Delete(context.Background(), ns); err != nil && !apierrors.IsNotFound(err) {
			tb.Errorf("delete test namespace %q: %v", name, err)
		}
	})
	return &TestEnv{tb: tb, rt: rt, namespace: name, opts: opts}
}

func (e *TestEnv) Namespace() string {
	return e.namespace
}

func (e *TestEnv) Client() client.Client {
	return e.rt.client
}

func (e *TestEnv) RESTConfig() *rest.Config {
	return e.rt.config
}

func (e *TestEnv) OperatorConfig() *configv1alpha1.OperatorConfiguration {
	return e.rt.operatorCfg
}

func (e *TestEnv) RuntimeConfig() *commoncontroller.RuntimeConfig {
	return e.rt.runtimeConfig
}

// ScaleClient returns a scale client configured for this envtest API server.
func (e *TestEnv) ScaleClient() (scale.ScalesGetter, error) {
	return newScaleClient(e.rt.config)
}

func (e *TestEnv) StartManager(setup func(ctrl.Manager) error) {
	e.tb.Helper()
	mgr, err := ctrl.NewManager(e.rt.config, ctrl.Options{
		Scheme:  e.rt.scheme,
		Metrics: metricsserver.Options{BindAddress: "0"},
		Cache: cache.Options{
			DefaultNamespaces: map[string]cache.Config{
				e.namespace: {},
			},
		},
	})
	if err != nil {
		e.tb.Fatalf("create manager: %v", err)
	}
	if err := setup(mgr); err != nil {
		e.tb.Fatalf("setup manager: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- mgr.Start(ctx)
	}()
	syncCtx, syncCancel := context.WithTimeout(context.Background(), e.opts.EventuallyTimeout)
	defer syncCancel()
	if !mgr.GetCache().WaitForCacheSync(syncCtx) {
		cancel()
		<-done
		e.tb.Fatal("manager cache did not sync")
	}
	e.tb.Cleanup(func() {
		cancel()
		if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
			e.tb.Errorf("manager stopped with error: %v", err)
		}
	})
}

func normalizeOptions(opts Options) Options {
	if !opts.Admission && !opts.Conversion {
		opts.Admission = true
		opts.Conversion = true
	}
	if opts.OperatorVersion == "" {
		opts.OperatorVersion = "1.0.0"
	}
	if opts.EventuallyTimeout == 0 {
		opts.EventuallyTimeout = 10 * time.Second
	}
	return opts
}

func defaultOperatorConfig(in *configv1alpha1.OperatorConfiguration) *configv1alpha1.OperatorConfiguration {
	cfg := &configv1alpha1.OperatorConfiguration{}
	if in != nil {
		*cfg = *in.DeepCopy()
	}
	configv1alpha1.SetDefaultsOperatorConfiguration(cfg)
	cfg.Server.Metrics.BindAddress = "0"
	cfg.Server.HealthProbe.BindAddress = "0"
	cfg.Server.Webhook.CertProvisionMode = configv1alpha1.CertProvisionModeManual
	cfg.GPU.DiscoveryEnabled = ptr.To(false)
	return cfg
}

func newScaleClient(config *rest.Config) (scale.ScalesGetter, error) {
	kubeClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, err
	}
	cachedDiscovery := memory.NewMemCacheClient(kubeClient.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscovery)
	return scale.NewForConfig(
		config,
		restMapper,
		dynamic.LegacyAPIPathResolverFunc,
		scale.NewDiscoveryScaleKindResolver(cachedDiscovery),
	)
}

func crdDirectoryPaths(opts Options) []string {
	if len(opts.CRDDirectoryPaths) > 0 {
		return opts.CRDDirectoryPaths
	}
	root := operatorRoot()
	return []string{
		filepath.Join(root, "config", "crd", "bases"),
		filepath.Join(root, "internal", "controller", "testing", "prometheus"),
		filepath.Join(root, "internal", "controller", "testing", "volcano.sh"),
		filepath.Join(root, "internal", "controller", "testing", "run.ai"),
		filepath.Join(root, "internal", "controller", "testing", "nvidia"),
	}
}

func binaryAssetsDirectory(opts Options) string {
	if opts.BinaryAssetsDirectory != "" {
		return opts.BinaryAssetsDirectory
	}
	if assets := os.Getenv("KUBEBUILDER_ASSETS"); assets != "" {
		return assets
	}
	return filepath.Join(operatorRoot(), "bin", "k8s", fmt.Sprintf("1.30.0-%s-%s", goruntime.GOOS, goruntime.GOARCH))
}

func operatorRoot() string {
	_, file, _, ok := goruntime.Caller(0)
	if !ok {
		panic("operatorenv: runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
}

func waitForWebhookServer(ctx context.Context, server webhook.Server) error {
	waitCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	return wait.PollUntilContextTimeout(waitCtx, 50*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
		err := server.StartedChecker()((*http.Request)(nil))
		return err == nil, nil
	})
}
