/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	nvidiav1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

func TestDGDRAdmissionDefaultsPlannerImage(t *testing.T) {
	ctx := context.Background()
	env := sharedEnv.ForTest(t)

	t.Log("Create a DGDR without spec.image through the API server")
	dgdr := &nvidiav1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "default-image",
			Namespace: env.Namespace(),
		},
		Spec: nvidiav1beta1.DynamoGraphDeploymentRequestSpec{
			Model:   "Qwen/Qwen3-0.6B",
			Backend: nvidiav1beta1.BackendTypeVllm,
			Hardware: &nvidiav1beta1.HardwareSpec{
				GPUSKU:         nvidiav1beta1.GPUSKUTypeH100SXM,
				VRAMMB:         ptr.To(81920.0),
				NumGPUsPerNode: ptr.To[int32](8),
				TotalGPUs:      ptr.To[int32](8),
			},
		},
	}
	if err := env.Client().Create(ctx, dgdr); err != nil {
		t.Fatalf("create DGDR: %v", err)
	}

	t.Log("Read back the DGDR and assert the admission webhook defaulted the planner image")
	var got nvidiav1beta1.DynamoGraphDeploymentRequest
	key := types.NamespacedName{Name: dgdr.Name, Namespace: env.Namespace()}
	if err := env.Client().Get(ctx, key, &got); err != nil {
		t.Fatalf("get DGDR: %v", err)
	}
	want := "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.0"
	if got.Spec.Image != want {
		t.Fatalf("defaulted image = %q, want %q", got.Spec.Image, want)
	}
}
