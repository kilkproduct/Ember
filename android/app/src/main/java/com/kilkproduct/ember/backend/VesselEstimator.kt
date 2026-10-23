package com.kilkproduct.ember.backend

/** Pure, immediate allocation arithmetic; no engine/RPC or device-fit promise.
 * Must match ember.trainer.estimate_vessel. Model/gradient/Adam: 16 B/parameter;
 * activations include explicit FFN width, quadratic attention and vocabulary logits.
 */
object VesselEstimator {
    fun estimate(config: ModelConfig, batch: Int = 2): VesselEstimate {
        val c=config.nEmbd.toLong();val layers=config.nLayer.toLong();val heads=config.nHead.toLong()
        val ctx=config.blockSize.toLong();val kv=config.nKvHead.toLong()
        require(batch in 1..32 && layers in 1..12 && c in 32..512 && heads in 1..16 && kv>0 && heads%kv==0L && c%heads==0L)
        require(ctx in 32..2048 && config.vocabSize in 261..50000 && config.ffnMult.isFinite() && config.ffnMult>0 && config.ffnMult<=16)
        val f=config.intermediateSize?.toLong() ?: kotlin.math.round(c*config.ffnMult).toLong()
        require(f in 1..8192)
        val params=config.vocabSize*c+c+layers*(2*c*c+2*c*(c/heads*kv)+3*c*f+2*c)
        val memory=params*16+batch*layers*(ctx*(32*c+12*f)+heads*ctx*ctx*16)+batch*ctx*config.vocabSize*16L
        val blocked=memory>512L*1024*1024
        val warning=when {
            blocked -> "Training estimate exceeds 512 MiB safety budget; reduce batch/context/width."
            memory>160L*1024*1024 -> "Large training working set; use batch 1–2 and monitor device memory."
            else -> null
        }
        return VesselEstimate(params,memory,config.vocabSize,warning,blocked)
    }
}
