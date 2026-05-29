package models

import "testing"

func TestGetBatchRelationClusterTool_NoError(t *testing.T) {
	tool, err := GetBatchRelationClusterTool()
	if err != nil {
		t.Fatalf("GetBatchRelationClusterTool() error: %v", err)
	}
	if tool.Function.Name != "batch_relation_cluster" {
		t.Errorf("tool name = %q, want %q", tool.Function.Name, "batch_relation_cluster")
	}
}
