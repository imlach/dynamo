/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package operatorenv

import (
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
)

const (
	dcdDefaultingPath                = "/mutate-nvidia-com-v1beta1-dynamocomponentdeployment"
	dgdDefaultingPath                = "/mutate/nvidia.com/v1beta1/dynamographdeployments"
	dgdrDefaultingPath               = "/mutate-nvidia-com-v1beta1-dynamographdeploymentrequest"
	podCheckpointRestoreMutationPath = "/mutate-core-v1-pod-checkpoint-restore"

	dcdValidationPath   = "/validate-nvidia-com-v1alpha1-dynamocomponentdeployment"
	dgdValidationPath   = "/validate/nvidia.com/v1beta1/dynamographdeployments"
	dckptValidationPath = "/validate-nvidia-com-v1alpha1-dynamocheckpoint"
	dmValidationPath    = "/validate-nvidia-com-v1alpha1-dynamomodel"
	dgdrValidationPath  = "/validate-nvidia-com-v1beta1-dynamographdeploymentrequest"
)

func webhookInstallOptions(opts Options) envtest.WebhookInstallOptions {
	install := envtest.WebhookInstallOptions{}
	if !opts.Admission {
		return install
	}
	install.MutatingWebhooks = []*admissionregistrationv1.MutatingWebhookConfiguration{
		mutatingWebhook("operatorenv-dcd-defaulting", dcdDefaultingPath, []admissionregistrationv1.OperationType{
			admissionregistrationv1.Create,
		}, "nvidia.com", []string{"v1beta1"}, "dynamocomponentdeployments"),
		mutatingWebhook("operatorenv-dgd-defaulting", dgdDefaultingPath, []admissionregistrationv1.OperationType{
			admissionregistrationv1.Create,
			admissionregistrationv1.Update,
		}, "nvidia.com", []string{"v1beta1"}, "dynamographdeployments"),
		mutatingWebhook("operatorenv-dgdr-defaulting", dgdrDefaultingPath, []admissionregistrationv1.OperationType{
			admissionregistrationv1.Create,
		}, "nvidia.com", []string{"v1beta1"}, "dynamographdeploymentrequests"),
		mutatingWebhook("operatorenv-pod-checkpoint-restore", podCheckpointRestoreMutationPath, []admissionregistrationv1.OperationType{
			admissionregistrationv1.Create,
		}, "", []string{"v1"}, "pods"),
	}
	install.ValidatingWebhooks = []*admissionregistrationv1.ValidatingWebhookConfiguration{
		validatingWebhook("operatorenv-dcd-validation", dcdValidationPath, "nvidia.com", []string{"v1alpha1"}, "dynamocomponentdeployments"),
		validatingWebhook("operatorenv-dgd-validation", dgdValidationPath, "nvidia.com", []string{"v1beta1"}, "dynamographdeployments"),
		validatingWebhook("operatorenv-dckpt-validation", dckptValidationPath, "nvidia.com", []string{"v1alpha1"}, "dynamocheckpoints"),
		validatingWebhook("operatorenv-dm-validation", dmValidationPath, "nvidia.com", []string{"v1alpha1"}, "dynamomodels"),
		validatingWebhook("operatorenv-dgdr-validation", dgdrValidationPath, "nvidia.com", []string{"v1beta1"}, "dynamographdeploymentrequests"),
	}
	return install
}

func mutatingWebhook(name, path string, operations []admissionregistrationv1.OperationType, group string, versions []string, resource string) *admissionregistrationv1.MutatingWebhookConfiguration {
	return &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: webhookObjectMeta(name),
		Webhooks: []admissionregistrationv1.MutatingWebhook{{
			Name:                    name + ".nvidia.com",
			ClientConfig:            webhookClientConfig(path),
			Rules:                   webhookRules(operations, group, versions, resource),
			FailurePolicy:           ptr.To(admissionregistrationv1.Fail),
			SideEffects:             ptr.To(admissionregistrationv1.SideEffectClassNone),
			AdmissionReviewVersions: []string{"v1"},
		}},
	}
}

func validatingWebhook(name, path string, group string, versions []string, resource string) *admissionregistrationv1.ValidatingWebhookConfiguration {
	return &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: webhookObjectMeta(name),
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name:         name + ".nvidia.com",
			ClientConfig: webhookClientConfig(path),
			Rules: webhookRules([]admissionregistrationv1.OperationType{
				admissionregistrationv1.Create,
				admissionregistrationv1.Update,
				admissionregistrationv1.Delete,
			}, group, versions, resource),
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
			Name:      "operatorenv-webhook",
			Path:      ptr.To(path),
		},
	}
}

func webhookRules(operations []admissionregistrationv1.OperationType, group string, versions []string, resource string) []admissionregistrationv1.RuleWithOperations {
	return []admissionregistrationv1.RuleWithOperations{{
		Operations: operations,
		Rule: admissionregistrationv1.Rule{
			APIGroups:   []string{group},
			APIVersions: versions,
			Resources:   []string{resource},
		},
	}}
}
