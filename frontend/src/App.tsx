import { useState, useCallback } from "react"
import { Upload } from "lucide-react"
import { Sidebar } from "@/components/layout/Sidebar"
import { Header } from "@/components/layout/Header"
import { FileUpload } from "@/components/upload/FileUpload"
import { QualityScoreCard } from "@/components/dashboard/QualityScoreCard"
import { DimensionChart } from "@/components/dashboard/DimensionChart"
import { DatasetStats } from "@/components/dashboard/DatasetStats"
import { AnomalySummary } from "@/components/anomaly/AnomalySummary"
import { HighRiskColumns } from "@/components/anomaly/HighRiskColumns"
import { OutlierTable } from "@/components/anomaly/OutlierTable"
import { ValidationErrors } from "@/components/anomaly/ValidationErrors"
import { AIAnalysisSummary } from "@/components/ai/AIAnalysisSummary"
import { LLMInsightsPanel } from "@/components/insights/LLMInsightsPanel"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { AnalysisStepper } from "@/components/upload/AnalysisStepper"
import { analyzeTransactions, ValidationFailureError } from "@/lib/api"
import type { AnalysisResponse } from "@/types/api"

function App() {
  const [activeSection, setActiveSection] = useState("upload")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [validationErrors, setValidationErrors] = useState<any[]>([])
  const [recommendation, setRecommendation] = useState<string | null>(null)
  const [data, setData] = useState<AnalysisResponse | null>(null)
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)
  const [analysisStep, setAnalysisStep] = useState(0)

  const handleFileSelect = useCallback(async (file: File) => {
    setIsLoading(true)
    setError(null)
    setActiveSection("analyzing")
    setAnalysisStep(1) // Move to "Quality Check" step (0-indexed logic in stepper might need adjustment, assume 0=step1, 1=step2)

    try {
      // Start the analysis
      const resultPromise = analyzeTransactions(file)

      // Artificial delay to show the "Analysis" step smoothly if the API is too fast, 
      // or just to give user context. 
      // In a real app we might get progress updates. 
      // For now, let's just wait for the promise.
      const result = await resultPromise

      setAnalysisStep(2) // Move to "Finalizing"

      // Short delay to let user see "Finalizing" state
      await new Promise(resolve => setTimeout(resolve, 800))

      setData(result)
      setActiveSection("overview")
    } catch (err) {
      setActiveSection("upload") // Go back to upload on error
      if (err instanceof ValidationFailureError) {
        setError(err.message)
        setValidationErrors(err.failures)
        setRecommendation(err.recommendation || null)
      } else {
        setError(err instanceof Error ? err.message : "Failed to analyze file")
        setValidationErrors([])
        setRecommendation(null)
      }
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Fallback UI for pages that require data
  const NoDataFallback = () => (
    <div className="flex flex-col items-center justify-center min-h-[50vh] text-center">
      <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mb-6">
        <Upload className="w-10 h-10 text-muted-foreground" />
      </div>
      <h2 className="text-2xl font-bold mb-2">No Data Available</h2>
      <p className="text-muted-foreground mb-6">
        Please upload a CSV file first to view this section.
      </p>
      <button
        onClick={() => setActiveSection("upload")}
        className="px-6 py-3 bg-primary text-primary-foreground rounded-xl font-medium hover:opacity-90 transition-opacity"
      >
        Go to Upload
      </button>
    </div>
  )

  const renderContent = () => {
    switch (activeSection) {
      case "upload":
        return (
          <div className="max-w-2xl mx-auto space-y-6">
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold mb-2 gradient-text">
                Transaction Data Analysis
              </h1>
              <p className="text-muted-foreground">
                Upload your CSV file for AI-powered data quality analysis
              </p>
            </div>

            <FileUpload onFileSelect={handleFileSelect} isLoading={isLoading} />

            {error && (
              <Alert variant="destructive" className="animate-in fade-in slide-in-from-top-2">
                <AlertTitle className="flex items-center gap-2 font-bold">
                  Error Analysis Failed
                </AlertTitle>
                <AlertDescription className="mt-2">
                  <p className="text-base font-medium mb-3">{error}</p>

                  {recommendation && (
                    <div className="mb-4 text-sm bg-destructive-foreground/10 p-3 rounded-md border border-destructive-foreground/20">
                      <span className="font-bold flex items-center gap-2 mb-1">
                        💡 Recommendation
                      </span>
                      {recommendation}
                    </div>
                  )}

                  {validationErrors.length > 0 && (
                    <div className="space-y-2 bg-white/90 dark:bg-black/40 p-3 rounded-md text-destructive-foreground">
                      <p className="font-semibold text-sm border-b border-destructive-foreground/20 pb-1 mb-2">
                        Critical Validation Failures ({validationErrors.length})
                      </p>
                      <ul className="list-disc pl-5 space-y-2 text-sm">
                        {validationErrors.map((fail, idx) => (
                          <li key={idx}>
                            <span className="font-semibold">Column `{fail.column}`</span>: {fail.message}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            )}
          </div>
        )

      case "overview":
        if (!data) return <NoDataFallback />
        return (
          <div className="space-y-6 animate-fade-in">
            <QualityScoreCard
              score={data.eda_analysis.data_quality_dimensions.overall_score}
              grade={data.eda_analysis.data_quality_dimensions.quality_grade}
              confidenceLevel={data.eda_analysis.data_quality_dimensions.confidence_level}
            />

            <DatasetStats
              rows={data.eda_analysis.shape.rows}
              columns={data.eda_analysis.shape.columns}
              missingColumnsCount={data.eda_analysis.missing_values.total_columns_with_missing}
              duplicatesCount={data.eda_analysis.duplicates.count}
              duplicatesPercentage={data.eda_analysis.duplicates.percentage}
            />

            <AnomalySummary summary={data.anomaly_report.summary} />
          </div>
        )

      case "dimensions":
        if (!data) return <NoDataFallback />
        return (
          <div className="space-y-6 animate-fade-in">
            <DimensionChart
              scores={data.eda_analysis.data_quality_dimensions.scores}
              chartType="bar"
            />
            <DimensionChart
              scores={data.eda_analysis.data_quality_dimensions.scores}
              chartType="radar"
            />
          </div>
        )

      case "anomalies":
        if (!data) return <NoDataFallback />
        return (
          <div className="space-y-6 animate-fade-in">
            <AnomalySummary summary={data.anomaly_report.summary} />

            <HighRiskColumns columns={data.anomaly_report.high_risk_columns} />


            <div className={`grid gap-6 ${data.anomaly_report.type_validation_errors.total_errors > 0 ? 'lg:grid-cols-2' : 'lg:grid-cols-1'}`}>
              <OutlierTable
                outliers={data.anomaly_report.ml_detected_outliers.by_column}
                totalOutliers={data.anomaly_report.ml_detected_outliers.total_outliers}
              />
              {data.anomaly_report.type_validation_errors.total_errors > 0 && (
                <ValidationErrors
                  errors={data.anomaly_report.type_validation_errors.errors}
                  totalErrors={data.anomaly_report.type_validation_errors.total_errors}
                />
              )}
            </div>
          </div>
        )

      case "ai-analysis":
        if (!data) return <NoDataFallback />
        return (
          <div className="space-y-6 animate-fade-in">
            <AIAnalysisSummary analysis={data.ai_analysis} />
          </div>
        )

      case "insights":
        if (!data) return <NoDataFallback />
        return (
          <div className="space-y-6 animate-fade-in">
            <LLMInsightsPanel insights={data.eda_analysis.data_quality_dimensions.llm_insights} />
          </div>
        )

      case "analyzing":
        return (
          <div className="flex flex-col items-center justify-center min-h-[60vh] animate-in fade-in zoom-in-95 duration-500">
            <AnalysisStepper currentStep={analysisStep} />
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Desktop Sidebar */}
      <div className="hidden lg:block">
        <Sidebar
          activeSection={activeSection}
          onSectionChange={setActiveSection}
          hasData={data !== null}
        />
      </div>

      {/* Mobile Sidebar Overlay */}
      {isMobileSidebarOpen && (
        <div
          className="fixed inset-0 z-50 lg:hidden"
          onClick={() => setIsMobileSidebarOpen(false)}
        >
          <div className="absolute inset-0 bg-black/50" />
          <div
            className="absolute left-0 top-0 h-full"
            onClick={e => e.stopPropagation()}
          >
            <Sidebar
              activeSection={activeSection}
              onSectionChange={(section) => {
                setActiveSection(section)
                setIsMobileSidebarOpen(false)
              }}
              hasData={data !== null}
            />
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="lg:ml-64">
        <Header
          datasetName={data?.dataset_name}
          timestamp={data?.timestamp}
          hasValidationErrors={data?.has_validation_errors}
          onMenuToggle={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
        />

        <main className="p-6">
          {renderContent()}
        </main>
      </div>
    </div>
  )
}

export default App
