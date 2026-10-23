package com.kilkproduct.ember.backend

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

/**
 * Regression tests for the staged-import 404.
 *
 * The engine fetches a picked file from `/__ember_file__/<token>`. The
 * token was once parsed as the first path segment, so the full URL path
 * passed by `shouldInterceptRequest` yielded `"__ember_file__"` instead of
 * the token, no staged file was ever found, and every import (Parquet
 * included) failed with "The selected file is no longer available
 * (HTTP 404)".
 */
class ImportTokenFromPathTest {

    // Tokens are dashless UUIDs minted in EmberEngine.stageImport.
    private val token = "0123456789abcdef0123456789abcdef"

    @Test
    fun `full URL path from shouldInterceptRequest yields the token`() {
        assertEquals(token, importTokenFromPath("/__ember_file__/$token"))
    }

    @Test
    fun `suffix path from PathHandler yields the token`() {
        assertEquals(token, importTokenFromPath(token))
    }

    @Test
    fun `leading slash suffix yields the token`() {
        assertEquals(token, importTokenFromPath("/$token"))
    }

    @Test
    fun `query and fragment are ignored`() {
        assertEquals(token, importTokenFromPath("/__ember_file__/$token?v=1"))
        assertEquals(token, importTokenFromPath("/__ember_file__/$token#frag"))
    }

    @Test
    fun `trailing slash is tolerated`() {
        assertEquals(token, importTokenFromPath("/__ember_file__/$token/"))
    }

    @Test
    fun `empty and prefix-only paths never yield a usable token`() {
        assertEquals("", importTokenFromPath(""))
        assertEquals("", importTokenFromPath("/"))
        // "/__ember_file__/" has no token at all. Whatever segment is left
        // must fail importFileFor's alphanumeric check, i.e. still 404
        // rather than accidentally matching a staged file.
        val leftover = importTokenFromPath("/__ember_file__/")
        assertFalse(leftover.isNotEmpty() && leftover.all { it.isLetterOrDigit() })
    }

    @Test
    fun `the prefix itself is never mistaken for a token with a real token present`() {
        // The exact failure mode of the old first-segment parse.
        assertFalse(importTokenFromPath("/__ember_file__/$token") == "__ember_file__")
    }
}
