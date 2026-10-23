package com.kilkproduct.ember

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import org.junit.Rule
import org.junit.Test

/** Device-required smoke test. Compilation is a CI gate; execution is separate. */
class NavigationSmokeTest {
    @get:Rule val compose=createAndroidComposeRule<MainActivity>()
    @Test fun chatDraftSurvivesTabSwitchAndActivityRecreation() {
        compose.onAllNodesWithText("Chat")[0].assertExists()
        compose.onAllNodesWithText("Prompt").filter(hasSetTextAction())[0].performTextInput("Local draft survives navigation")
        compose.onAllNodesWithText("Data")[0].performClick()
        compose.onAllNodesWithText("Chat")[0].performClick()
        compose.onAllNodesWithText("Prompt").filter(hasSetTextAction())[0].assertTextContains("Local draft survives navigation")
        compose.activityRule.scenario.recreate()
        compose.onAllNodesWithText("Prompt").filter(hasSetTextAction())[0].assertTextContains("Local draft survives navigation")
    }
}
