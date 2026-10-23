package com.kilkproduct.ember.backend

import com.kilkproduct.ember.MainViewModel
import org.junit.Assert.*
import org.junit.Test

class OverhaulContractTest {
    @Test fun expandedConfigRoundTripsAllPreparationFields() {
        val cfg=ModelConfig(3,4,128,256,50000,0.0,4.0,nKvHead=2,ropeTheta=10000.0,
            maxTrainTokens=500000,maxValTokens=16000,tokenizerChars=123456,intermediateSize=384,seed=123)
        assertEquals(cfg,ModelConfig.fromJson(cfg.toJson()))
    }
    @Test fun rollingEtaAndRealForageCountersParse() {
        val progress=TrainingProgress.fromJson(org.json.JSONObject().put("etaSeconds",12.5))
        assertEquals(12.5,progress.etaSeconds!!,0.0)
        assertNull(TrainingProgress.fromJson(org.json.JSONObject().put("etaSeconds",org.json.JSONObject.NULL)).etaSeconds)
        val f=ForagingSnapshot.fromJson(org.json.JSONObject().put("cached",3).put("bytesDownloaded",1000)
            .put("autosave",org.json.JSONObject().put("pendingTopics",org.json.JSONArray().put("topic"))))
        assertEquals(3L,f.cached);assertEquals(1,f.pendingTopics);assertEquals(1000L,f.bytesDownloaded)
    }
    @Test fun allPresetsHaveValidShapesAndMonotonicWorkingSets() {
        assertTrue(MainViewModel.VESSEL_PRESETS.size>=10)
        for((_,p) in MainViewModel.VESSEL_PRESETS) {
            val cfg=ModelConfig(p.layers,p.heads,p.embd,p.ctx,p.vocab,0.0,p.ffn.toDouble(),nKvHead=p.kv)
            val e=VesselEstimator.estimate(cfg)
            assertTrue(e.params>0);assertTrue(e.trainBytes>e.params*16)
            assertTrue(VesselEstimator.estimate(cfg,4).trainBytes>e.trainBytes)
        }
    }
    @Test fun compressedImportNeverUsesCompressedSizeAsDecompressedTotal() {
        val p=ImportProgress.fromJson(org.json.JSONObject().put("total",0).put("records",100)
            .put("documents",90).put("byteBasis","decompressed").put("bytesRead",100000).put("sourceBytes",1000))
        assertEquals(0,p.total);assertTrue(p.progressFraction.isNaN())
        assertTrue(p.stageLabel.isNotBlank())
    }
    @Test fun largeVocabAndExplicitFFNIncreaseRealEstimate() {
        val cfg=ModelConfig(2,2,64,64,1500,0.0,4.0)
        val e=VesselEstimator.estimate(cfg)
        assertTrue(VesselEstimator.estimate(cfg.copy(vocabSize=261)).params>0)
        assertEquals(219200L,e.params) // pinned against Python's tensor_specs
        assertTrue(VesselEstimator.estimate(cfg.copy(vocabSize=50000)).trainBytes>e.trainBytes)
        assertTrue(VesselEstimator.estimate(cfg.copy(intermediateSize=512)).trainBytes>e.trainBytes)
    }
}
