// static/js/script.js
import { fetchPredictionData } from './api.js'; // Import the API function

// --- DOM Element References ---
const predictButton = document.getElementById('predictButton'); //
const gaugeFill = document.getElementById('gaugeFill'); //
const gaugeText = document.getElementById('gaugeText'); //
const thresholdMark = document.getElementById('thresholdMark'); //
const thresholdValue = document.getElementById('thresholdValue'); //
const clientInfoDiv = document.getElementById('clientInfoDiv'); //
const toggleDashboardCheckbox = document.getElementById('toggleDashboard'); //
const dashboardDiv = document.getElementById('dashboardDiv'); //
const toggleDisplayCheckbox = document.getElementById('toggleDisplay'); // Checkbox for display mode
const resultBox = document.getElementById('resultBox'); // Gauge container for border

// Keep track of the current prediction so we can re-render in place
let currentPrediction = null; //

// Helper function to get CSS variable values
function getCssVariable(variableName) {
  return getComputedStyle(document.documentElement).getPropertyValue(variableName).trim(); //
}

/**
 * Updates the gauge visualization with prediction results using CSS variables.
 * @param {number} threshold - The decision threshold (e.g., 0.5) TO DISPLAY ON GAUGE.
 * @param {number} probability - The chosen probability (0 to 1) to fill the gauge.
 * @param {string} displayText - The full text to display in the gauge (handles both cases).
 * @param {boolean} isAcceptedOrLowRisk - Determines the color scheme (true for accepted/low risk, false for refused/high risk)
 */
function updateGauge(threshold, probability, displayText, isAcceptedOrLowRisk) {
  probability = Math.max(0, Math.min(1, probability)); // Ensure probability is between 0 and 1
  threshold = Math.max(0, Math.min(1, threshold));     // Ensure threshold is between 0 and 1

  const fillWidth = (probability * 100).toFixed(2) + '%'; //
  gaugeFill.style.width = fillWidth; //

  let fillColorVar, textColorVar, borderColorVar;
  if (isAcceptedOrLowRisk) {
    fillColorVar = '--gauge-accepted-fill'; //
    textColorVar = '--gauge-accepted-text'; //
    borderColorVar = '--gauge-accepted-border'; //
  } else {
    fillColorVar = '--gauge-refused-fill'; //
    textColorVar = '--gauge-refused-text'; //
    borderColorVar = '--gauge-refused-border'; //
  }
  gaugeFill.style.backgroundColor = getCssVariable(fillColorVar); //
  gaugeText.style.color = getCssVariable(textColorVar); //
  resultBox.style.borderColor = getCssVariable(borderColorVar); //

  gaugeText.textContent = displayText; //

  const thresholdLeftPercent = (threshold * 100).toFixed(2) + '%'; //
  thresholdMark.style.left = `calc(${thresholdLeftPercent} - 1px)`; //
  thresholdValue.style.left = thresholdLeftPercent; //
  thresholdValue.textContent = threshold.toFixed(2); // Display threshold with 2 decimal places

  thresholdMark.style.visibility = 'visible'; //
  thresholdValue.style.visibility = 'visible'; //
 }


/**
 * Updates the client information display box.
 * @param {number} loanId - The client's loan ID.
 * @param {number} creditAmount - The requested credit amount.
 */
function updateClientInfo(loanId, creditAmount) {
    const amount = (creditAmount !== null && creditAmount !== undefined) ? creditAmount : 0; //
    const formattedAmount = amount.toLocaleString('en-US', { //
        style: 'currency', //
        currency: 'USD', //
        minimumFractionDigits: 0, //
        maximumFractionDigits: 0 //
    });

    if (clientInfoDiv) { // Check if the element exists
        clientInfoDiv.innerHTML = `
            <p class="crp-client-info__line"><strong>Loan ID:</strong> ${loanId !== null && loanId !== undefined ? loanId : 'N/A'}</p>
            <p class="crp-client-info__line"><strong>Credit Amount:</strong> ${formattedAmount}</p>
        `; //
    } else {
        console.error("updateClientInfo: clientInfoDiv is null!"); //
    }
}


/**
 * Decides how to display the current prediction based on the checkbox
 * and updates the gauge + client info accordingly.
 */
function renderPrediction() {
  if (!currentPrediction) {
    console.log("Render prediction called but no current prediction data."); //
    return; //
  }
  console.log("Rendering prediction with data:", currentPrediction); //

  const originalThreshold = currentPrediction.threshold; //
  const pNeg = currentPrediction.probability_neg ?? currentPrediction.probabilityClass0; // Probability of not defaulting
  const pPos = currentPrediction.probability_pos ?? currentPrediction.probabilityClass1; // Probability of defaulting
  const loanId = currentPrediction.loan_id; //
  const creditAmount = currentPrediction.credit_amount; //
  const decisionFromAPI = currentPrediction.decision; //

  if (originalThreshold === undefined || pNeg === undefined || pPos === undefined || decisionFromAPI === undefined) {
       console.error("Cannot render prediction: Missing threshold, probability, or decision values.", currentPrediction); //
       gaugeText.textContent = "Error: Incomplete prediction data"; //
       gaugeFill.style.backgroundColor = getCssVariable('--gauge-refused-fill'); //
       gaugeFill.style.width = '100%'; //
       resultBox.style.borderColor = getCssVariable('--gauge-refused-border'); //
       if (clientInfoDiv) {
           clientInfoDiv.innerHTML = '<p>Could not display prediction details.</p>'; //
       }
       return; //
  }

  let probabilityUsed;
  let displayText;
  let isAcceptedOrLowRisk;
  let thresholdToDisplayOnGauge; // The threshold value shown on the gauge UI

  if (toggleDisplayCheckbox.checked) {
    // --- End User Display (Checked) ---
    // User sees their "score" as pNeg (probability of acceptance / not defaulting).
    // The threshold is inverted: it's the minimum "acceptance score" needed.
    probabilityUsed = pNeg; //
    // thresholdForComparison = 1 - originalThreshold; // Kept for reference, not directly for decision text
    thresholdToDisplayOnGauge = 1 - originalThreshold; // Display this inverted threshold

    // Decision text is based on decisionFromAPI
    const userDisplayOutcomeText = decisionFromAPI === "accepted" ? "CREDIT ACCEPTED" : "CREDIT REFUSED";
    displayText = `${userDisplayOutcomeText} (score: ${probabilityUsed.toFixed(2)})`;
    isAcceptedOrLowRisk = (decisionFromAPI === "accepted");

    console.log(`End-user display: API Decision = ${decisionFromAPI}, Score (P(Neg)) = ${pNeg.toFixed(4)}, Gauge Threshold = ${thresholdToDisplayOnGauge.toFixed(4)}`);
    console.log(`   (Original Risk Threshold for P(Pos) was: ${originalThreshold.toFixed(4)})`); //

  } else {
    // --- Internal Risk Display (Unchecked) ---
    // We show pPos (probability of defaulting / risk score).
    // The threshold is the original risk threshold.
    probabilityUsed = pPos; //
    // thresholdForComparison = originalThreshold; // Kept for reference, not directly for decision text
    thresholdToDisplayOnGauge = originalThreshold; // Display this original threshold

    // Decision text is based on decisionFromAPI
    // "accepted" by the model means "LOW RISK" in this internal view
    const internalDisplayRiskText = decisionFromAPI === "accepted" ? "LOW RISK" : "HIGH RISK";
    displayText = `${internalDisplayRiskText} (score: ${probabilityUsed.toFixed(2)})`;
    isAcceptedOrLowRisk = (decisionFromAPI === "accepted"); // "accepted" from API means low risk

    console.log(`Internal display: API Decision = ${decisionFromAPI}, Risk Score (P(Pos)) = ${pPos.toFixed(4)}, Gauge Threshold = ${thresholdToDisplayOnGauge.toFixed(4)}`);
  }

  // Update the gauge with the probability used and the threshold to display on the gauge
  updateGauge(thresholdToDisplayOnGauge, probabilityUsed, displayText, isAcceptedOrLowRisk); //
  updateClientInfo(loanId, creditAmount); //
}


/**
 * Handles the prediction button click: calls the API and updates UI.
 */
async function handlePrediction() {
  predictButton.disabled = true; //
  predictButton.textContent = 'Predicting...'; //

  gaugeText.textContent = "Loading prediction..."; //
  gaugeFill.style.width = '0%'; //
  gaugeFill.style.backgroundColor = 'transparent'; //
  resultBox.style.borderColor = getCssVariable('--border-color'); //
  if (clientInfoDiv) { // Check if element exists before setting innerHTML
    clientInfoDiv.innerHTML = '<p>Loading client info...</p>'; //
  }
  thresholdMark.style.visibility = 'hidden'; //
  thresholdValue.style.visibility = 'hidden'; //

  try {
    console.log("Script: Calling fetchPredictionData..."); //
    currentPrediction = await fetchPredictionData(); //
    renderPrediction(); //

  } catch (error) {
    console.error('Script: Error fetching or processing prediction:', error); //
    gaugeText.textContent = `Error: ${error.message || "Prediction failed"}`; //
    gaugeFill.style.backgroundColor = getCssVariable('--gauge-refused-fill'); //
    gaugeFill.style.width = '100%'; //
    resultBox.style.borderColor = getCssVariable('--gauge-refused-border'); //
    if (clientInfoDiv) { // Check if element exists
        clientInfoDiv.innerHTML = '<p>Could not retrieve prediction details.</p>'; //
    }
    currentPrediction = null; //
  } finally {
    predictButton.disabled = false; //
    predictButton.textContent = 'Predict Next Client'; //
  }
}

/**
 * Toggles the visibility of the enhanced dashboard section.
 */
function toggleDashboard() {
    console.log("CALLED: toggleDashboard function."); //
    console.log("  > toggleDashboardCheckbox element:", toggleDashboardCheckbox); //
    console.log("  > dashboardDiv element:", dashboardDiv); //

    if (!toggleDashboardCheckbox || !dashboardDiv) {
        console.error("ERROR inside toggleDashboard: toggleDashboardCheckbox or dashboardDiv is null. Cannot change display."); //
        return; //
    }
    dashboardDiv.style.display = toggleDashboardCheckbox.checked ? 'block' : 'none'; //
    console.log(`INFO: dashboardDiv (id='${dashboardDiv.id}') display set to: ${dashboardDiv.style.display} (checkbox checked: ${toggleDashboardCheckbox.checked})`); //
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
  console.log("DOM Content Loaded. Setting up event listeners."); //

  if (predictButton) {
    predictButton.addEventListener('click', handlePrediction); //
  } else {
    console.error("Predict button (id='predictButton') not found!"); //
  }

  // For "Toggle Enhanced Dashboard"
  if (toggleDashboardCheckbox) {
    console.log("SUCCESS: Found toggleDashboardCheckbox element:", toggleDashboardCheckbox); //
    if (dashboardDiv) {
      console.log("SUCCESS: Found dashboardDiv element:", dashboardDiv); //
      toggleDashboardCheckbox.addEventListener('change', toggleDashboard); //
      console.log("SUCCESS: Event listener attached to toggleDashboardCheckbox."); //
      toggleDashboard(); // Initial call to set dashboard state based on checkbox
    } else {
      console.error("ERROR: dashboardDiv (id='dashboardDiv') NOT found. Dashboard toggle will not work."); //
    }
  } else {
    console.error("ERROR: toggleDashboardCheckbox (id='toggleDashboard') NOT found. Check HTML and script.js."); //
  }

  // For "End User Display"
  if (toggleDisplayCheckbox) {
    console.log("SUCCESS: Found toggleDisplayCheckbox element:", toggleDisplayCheckbox); //
    toggleDisplayCheckbox.addEventListener('change', renderPrediction); //
    console.log("SUCCESS: Event listener attached to toggleDisplayCheckbox."); //
    if (currentPrediction) renderPrediction(); //
  } else {
    console.error("ERROR: End User Display checkbox (id='toggleDisplay') not found!"); //
  }
});
