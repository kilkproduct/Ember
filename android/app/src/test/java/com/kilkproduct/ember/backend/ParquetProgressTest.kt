package com.kilkproduct.ember.backend

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ParquetProgressTest {
    @Test fun persistedRowsAndProjectedReadCountersRemainDistinct() {
        val progress=ImportProgress.fromJson(JSONObject()
            .put("token","shard").put("documents",90).put("records",100)
            .put("total",250).put("percent",40).put("rejected",10)
            .put("rowGroupsProcessed",1).put("rowGroups",3)
            .put("bytesRead",1234L).put("sourceBytes",9000L)
            .put("byteBasis","projected pages and footer (includes resume replay)")
            .put("documentsPerSecond",35.5).put("stage","running"))
        assertEquals(90,progress.processed)
        assertEquals(100,progress.records)
        assertEquals(10,progress.rejected)
        assertEquals(1,progress.rowGroupsProcessed)
        assertEquals(3,progress.rowGroups)
        assertEquals(35.5,progress.documentsPerSecond,0.001)
        assertEquals(0.4f,progress.progressFraction,0.001f)
    }
}
