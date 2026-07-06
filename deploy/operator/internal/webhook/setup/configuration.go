/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package setup

import (
	webhookdefaulting "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/defaulting"
	webhookmutation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/mutation"
	webhookvalidation "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/validation"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

const (
	nvidiaAPIGroup = "nvidia.com"
	nvidiaV1Alpha1 = "v1alpha1"
	nvidiaV1Beta1  = "v1beta1"
)

// AdmissionWebhooks returns configurations for the production admission endpoints registered by SetupAll.
func AdmissionWebhooks() (
	[]*admissionregistrationv1.MutatingWebhookConfiguration,
	[]*admissionregistrationv1.ValidatingWebhookConfiguration,
) {
	return []*admissionregistrationv1.MutatingWebhookConfiguration{
			mutatingWebhook(
				"dynamo-operator-dcd-defaulting",
				webhookdefaulting.DCDDefaultingWebhookPath,
				[]admissionregistrationv1.OperationType{
					admissionregistrationv1.Create,
				},
				nvidiaAPIGroup,
				nvidiaV1Beta1,
				"dynamocomponentdeployments",
			),
			mutatingWebhook(
				"dynamo-operator-dgd-defaulting",
				webhookdefaulting.DGDV1Beta1DefaultingWebhookPath,
				[]admissionregistrationv1.OperationType{
					admissionregistrationv1.Create,
					admissionregistrationv1.Update,
				},
				nvidiaAPIGroup,
				nvidiaV1Beta1,
				"dynamographdeployments",
			),
			mutatingWebhook(
				"dynamo-operator-dgd-v1alpha1-defaulting",
				webhookdefaulting.DGDV1Alpha1DefaultingWebhookPath,
				[]admissionregistrationv1.OperationType{
					admissionregistrationv1.Create,
					admissionregistrationv1.Update,
				},
				nvidiaAPIGroup,
				nvidiaV1Alpha1,
				"dynamographdeployments",
			),
			mutatingWebhook(
				"dynamo-operator-dgdr-defaulting",
				webhookdefaulting.DGDRDefaultingWebhookPath,
				[]admissionregistrationv1.OperationType{
					admissionregistrationv1.Create,
				},
				nvidiaAPIGroup,
				nvidiaV1Beta1,
				"dynamographdeploymentrequests",
			),
			mutatingWebhook(
				"dynamo-operator-pod-checkpoint-restore",
				webhookmutation.PodCheckpointRestoreWebhookPath,
				[]admissionregistrationv1.OperationType{
					admissionregistrationv1.Create,
				},
				"",
				"v1",
				"pods",
			),
		}, []*admissionregistrationv1.ValidatingWebhookConfiguration{
			validatingWebhook(
				"dynamo-operator-dcd-validation",
				webhookvalidation.DynamoComponentDeploymentWebhookPath,
				nvidiaV1Alpha1,
				"dynamocomponentdeployments",
			),
			validatingWebhook(
				"dynamo-operator-dgd-validation",
				webhookvalidation.DynamoGraphDeploymentV1Beta1WebhookPath,
				nvidiaV1Beta1,
				"dynamographdeployments",
			),
			validatingWebhook(
				"dynamo-operator-dgd-v1alpha1-validation",
				webhookvalidation.DynamoGraphDeploymentV1Alpha1WebhookPath,
				nvidiaV1Alpha1,
				"dynamographdeployments",
			),
			validatingWebhook(
				"dynamo-operator-dckpt-validation",
				webhookvalidation.DynamoCheckpointWebhookPath,
				nvidiaV1Alpha1,
				"dynamocheckpoints",
			),
			validatingWebhook(
				"dynamo-operator-dm-validation",
				webhookvalidation.DynamoModelWebhookPath,
				nvidiaV1Alpha1,
				"dynamomodels",
			),
			validatingWebhook(
				"dynamo-operator-dgdr-validation",
				webhookvalidation.DynamoGraphDeploymentRequestWebhookPath,
				nvidiaV1Beta1,
				"dynamographdeploymentrequests",
			),
		}
}

func mutatingWebhook(
	name, path string,
	operations []admissionregistrationv1.OperationType,
	group string,
	version string,
	resource string,
) *admissionregistrationv1.MutatingWebhookConfiguration {
	return &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: webhookObjectMeta(name),
		Webhooks: []admissionregistrationv1.MutatingWebhook{{
			Name:                    name + ".nvidia.com",
			ClientConfig:            webhookClientConfig(path),
			Rules:                   webhookRules(operations, group, version, resource),
			FailurePolicy:           ptr.To(admissionregistrationv1.Fail),
			SideEffects:             ptr.To(admissionregistrationv1.SideEffectClassNone),
			AdmissionReviewVersions: []string{"v1"},
		}},
	}
}

func validatingWebhook(
	name, path string,
	version string,
	resource string,
) *admissionregistrationv1.ValidatingWebhookConfiguration {
	return &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: webhookObjectMeta(name),
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name:         name + ".nvidia.com",
			ClientConfig: webhookClientConfig(path),
			Rules: webhookRules([]admissionregistrationv1.OperationType{
				admissionregistrationv1.Create,
				admissionregistrationv1.Update,
				admissionregistrationv1.Delete,
			}, nvidiaAPIGroup, version, resource),
			FailurePolicy:           ptr.To(admissionregistrationv1.Fail),
			SideEffects:             ptr.To(admissionregistrationv1.SideEffectClassNone),
			AdmissionReviewVersions: []string{"v1"},
		}},
	}
}

func webhookObjectMeta(name string) metav1.ObjectMeta {
	return metav1.ObjectMeta{Name: name}
}

func webhookClientConfig(path string) admissionregistrationv1.WebhookClientConfig {
	return admissionregistrationv1.WebhookClientConfig{
		Service: &admissionregistrationv1.ServiceReference{
			Namespace: "default",
			Name:      "dynamo-operator-webhook",
			Path:      ptr.To(path),
		},
	}
}

func webhookRules(
	operations []admissionregistrationv1.OperationType,
	group string,
	version string,
	resource string,
) []admissionregistrationv1.RuleWithOperations {
	return []admissionregistrationv1.RuleWithOperations{{
		Operations: operations,
		Rule: admissionregistrationv1.Rule{
			APIGroups:   []string{group},
			APIVersions: []string{version},
			Resources:   []string{resource},
		},
	}}
}
