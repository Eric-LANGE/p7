// static/js/api.js

/**
 * Fetches prediction data from the backend API.
 * @returns {Promise<object>} A promise that resolves with the prediction data object.
 * @throws {Error} If the API request fails or returns an error status.
 */
export async function fetchPredictionData() {
    console.log("API module: Fetching prediction data...");
    const response = await fetch('/predict');

    if (!response.ok) {
        let errorDetail = 'Unknown server error';
        try {
            // Try to parse error response from API
            const errorData = await response.json();
            errorDetail = errorData.detail || JSON.stringify(errorData);
        } catch (parseError) {
            // If parsing fails, use status text
            errorDetail = response.statusText;
            console.warn("API module: Could not parse error response body.");
        }
        console.error(`API module: Error response status ${response.status}: ${errorDetail}`);
        throw new Error(`API request failed: ${errorDetail} (Status: ${response.status})`);
    }

    try {
        const data = await response.json();
        console.log("API module: Prediction data received:", data);
        return data;
    } catch (jsonError) {
        console.error("API module: Failed to parse JSON response:", jsonError);
        throw new Error("Failed to parse prediction data from server.");
    }
}
