# JNI names are a stable native ABI. Chaquopy supplies its consumer keep rules.
-keep class com.kilkproduct.ember.backend.LlamaInference { *; }
-keep class com.kilkproduct.ember.backend.LlamaInference$* { *; }
-keep @androidx.annotation.Keep class * { *; }
-keepclassmembers class * { @androidx.annotation.Keep *; }
