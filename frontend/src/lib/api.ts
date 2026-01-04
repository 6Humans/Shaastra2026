import axios, { type AxiosError } from "axios";
import type { AnalysisResponse, APIError } from "@/types/api";

const API_BASE_URL = "http://localhost:8000";

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        "ngrok-skip-browser-warning": "true",
    },
});

export async function analyzeTransactions(
    file: File,
    numSamples: number = 5
): Promise<AnalysisResponse> {
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await apiClient.post<AnalysisResponse>(
            `/analyze-transactions?num_samples=${numSamples}`,
            formData,
            {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            }
        );
        return response.data;
    } catch (error) {
        const axiosError = error as AxiosError<APIError>;
        if (axiosError.response?.status === 422) {
            throw new Error(
                axiosError.response.data?.message ||
                "Critical validation failures detected"
            );
        }
        throw new Error(
            axiosError.response?.data?.message || "Failed to analyze transactions"
        );
    }
}

export async function healthCheck(): Promise<{
    status: string;
    timestamp: string;
}> {
    const response = await apiClient.get("/health");
    return response.data;
}
