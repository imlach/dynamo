# Operator Env Design

`operatorenv` provides a Kubernetes API server for controller tests. It runs
the production admission and conversion webhook registrations, but no
controllers until an individual test starts one.

## Lifecycle

`Env` owns either a shared envtest process or an isolated process:

```go
var sharedEnv = operatorenv.New(operatorenv.Options{
    Admission:  true,
    Conversion: true,
})

func TestMain(m *testing.M) {
    os.Exit(sharedEnv.RunM(m))
}

func TestSomething(t *testing.T) {
    env := sharedEnv.ForTest(t)
    // ...
}
```

`RunM` starts the shared API server lazily, on the first `ForTest` call, and
stops it after the package test run. Each `ForTest` creates and cleans up a
unique namespace. This keeps API-server startup cheap without sharing test
objects.

`RunT` starts an isolated API server for a test that cannot share a cluster:

```go
func TestIsolated(t *testing.T) {
    env := operatorenv.New(operatorenv.Options{}).RunT(t)
    // ...
}
```

## Webhooks

The envtest API server renders the production Helm webhook configuration and
installs its mutating and validating webhook objects when admission is enabled.
A dedicated webhook manager registers the production handlers via
`webhooksetup.SetupAll`, including conversion endpoints. Therefore normal
`Client` CRUD reaches the API server, CRD CEL validation, and production webhook
code.

The webhook manager is separate from controller managers. It runs for the
lifetime of the environment and is the only always-on manager.

## Controller Managers

Tests start only the controller they need:

```go
env.StartManager(func(mgr ctrl.Manager) error {
    return controller.SetupDynamoGraphDeployment(mgr, options)
})
```

`StartManager` limits its cache to the test namespace and stops the manager at
test cleanup. Controller setup functions belong to `internal/controller`; the
test package assembles their dependencies from `TestEnv` to avoid an import
cycle.

## Scope

`operatorenv` owns API-server, namespace, webhook, and controller-manager
lifecycle. It does not own YAML fixture loading or expected-manifest matching.
Those APIs should be introduced with the first controller test that uses them.
