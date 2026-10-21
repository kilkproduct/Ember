package com.kilkproduct.ember.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

val EmberOrange = Color(0xFFD96A33)
val EmberOrangeDeep = Color(0xFF8E4523)
val EmberTeal = Color(0xFF5CB6A8)
val EmberBackground = Color(0xFF12100E)
val EmberSurface = Color(0xFF191613)
val EmberSurfaceHigh = Color(0xFF221D18)
val EmberSurfaceHighest = Color(0xFF2A241E)
val EmberText = Color(0xFFEDE6DE)
val EmberMuted = Color(0xFFA79A8D)
val EmberOutline = Color(0xFF3B332B)
val EmberError = Color(0xFFE0776F)
val EmberSuccess = Color(0xFF8DBE7C)
val EmberWarning = Color(0xFFD9A441)

private val EmberColorScheme = darkColorScheme(
    primary = EmberOrange,
    onPrimary = Color(0xFF1D0F07),
    primaryContainer = Color(0xFF432413),
    onPrimaryContainer = Color(0xFFFFD9C3),
    secondary = EmberTeal,
    onSecondary = Color(0xFF06110F),
    secondaryContainer = Color(0xFF17322E),
    onSecondaryContainer = Color(0xFFC7F2EA),
    tertiary = EmberWarning,
    onTertiary = Color(0xFF1B1304),
    background = EmberBackground,
    onBackground = EmberText,
    surface = EmberSurface,
    onSurface = EmberText,
    surfaceVariant = EmberSurfaceHigh,
    onSurfaceVariant = EmberMuted,
    surfaceContainer = EmberSurface,
    surfaceContainerHigh = EmberSurfaceHigh,
    surfaceContainerHighest = EmberSurfaceHighest,
    outline = EmberOutline,
    outlineVariant = Color(0xFF2C251F),
    error = EmberError,
    onError = Color(0xFF2A0907),
    errorContainer = Color(0xFF3A1513),
    onErrorContainer = Color(0xFFFFDAD6),
    inverseSurface = EmberText,
    inverseOnSurface = EmberBackground,
    inversePrimary = EmberOrangeDeep,
)

private val EmberTypography = Typography(
    displaySmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 30.sp,
        lineHeight = 35.sp,
        letterSpacing = (-0.3).sp,
    ),
    headlineSmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 22.sp,
        lineHeight = 28.sp,
        letterSpacing = (-0.1).sp,
    ),
    titleLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.SemiBold,
        fontSize = 19.sp,
        lineHeight = 24.sp,
    ),
    titleMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 15.sp,
        lineHeight = 20.sp,
    ),
    titleSmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 13.sp,
        lineHeight = 18.sp,
    ),
    bodyLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontSize = 15.sp,
        lineHeight = 22.sp,
    ),
    bodyMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontSize = 14.sp,
        lineHeight = 20.sp,
    ),
    bodySmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontSize = 12.sp,
        lineHeight = 17.sp,
    ),
    labelLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 13.sp,
        lineHeight = 18.sp,
    ),
    labelMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 12.sp,
        lineHeight = 16.sp,
    ),
    labelSmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.Medium,
        fontSize = 11.sp,
        lineHeight = 15.sp,
        letterSpacing = 0.2.sp,
    ),
)

private val EmberShapes = Shapes(
    extraSmall = RoundedCornerShape(6.dp),
    small = RoundedCornerShape(8.dp),
    medium = RoundedCornerShape(12.dp),
    large = RoundedCornerShape(16.dp),
    extraLarge = RoundedCornerShape(20.dp),
)

@Composable
fun EmberTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = EmberColorScheme,
        typography = EmberTypography,
        shapes = EmberShapes,
        content = content,
    )
}
