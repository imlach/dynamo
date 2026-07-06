/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	nvidiav1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiav1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

const backendFrameworkVLLM = "vllm"

func TestDGDV1Beta1ConversionSmoke(t *testing.T) {
	ctx := context.Background()
	env := sharedEnv.ForTest(t)

	t.Log("Create a v1beta1 DGD through the API server")
	dgd := &nvidiav1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "conversion-dgd",
			Namespace: env.Namespace(),
		},
		Spec: nvidiav1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: backendFrameworkVLLM,
			Components: []nvidiav1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "frontend",
				ComponentType: nvidiav1beta1.ComponentTypeFrontend,
			}},
		},
	}
	if err := env.Client().Create(ctx, dgd); err != nil {
		t.Fatalf("create v1beta1 DGD: %v", err)
	}

	t.Log("Read the same DGD as v1alpha1 to exercise the conversion webhook")
	var alpha nvidiav1alpha1.DynamoGraphDeployment
	key := types.NamespacedName{Name: dgd.Name, Namespace: env.Namespace()}
	if err := env.Client().Get(ctx, key, &alpha); err != nil {
		t.Fatalf("get DGD as v1alpha1 through conversion webhook: %v", err)
	}

	t.Log("Assert the converted DGD uses the v1alpha1 service map shape")
	if alpha.Spec.BackendFramework != backendFrameworkVLLM {
		t.Fatalf("v1alpha1 backendFramework = %q, want %q", alpha.Spec.BackendFramework, backendFrameworkVLLM)
	}
	if alpha.Spec.Services["frontend"] == nil {
		t.Fatalf("v1alpha1 services missing converted frontend component: %#v", alpha.Spec.Services)
	}
}
