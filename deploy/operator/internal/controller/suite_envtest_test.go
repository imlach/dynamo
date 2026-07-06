/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 Atalaya Tech. Inc
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"os"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

var (
	sharedEnv = operatorenv.New(operatorenv.Options{
		Admission:       true,
		Conversion:      true,
		OperatorVersion: "1.1.0",
	})
	ctx              = context.Background()
	k8sClient        client.Client
	envtestNamespace string
)

func TestMain(m *testing.M) {
	os.Exit(sharedEnv.RunM(m))
}

func TestControllers(t *testing.T) {
	logf.SetLogger(zap.New(zap.WriteTo(GinkgoWriter), zap.UseDevMode(true)))
	RegisterFailHandler(Fail)
	RunSpecs(t, "Controller Suite")
}

var _ = BeforeEach(func() {
	env := sharedEnv.ForTest(GinkgoTB())
	k8sClient = env.Client()
	envtestNamespace = env.Namespace()
})
