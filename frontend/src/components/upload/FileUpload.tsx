import { useCallback, useState } from "react"
import { Upload, FileText, X, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

interface FileUploadProps {
    onFileSelect: (file: File) => void
    isLoading?: boolean
    accept?: string
}

export function FileUpload({
    onFileSelect,
    isLoading = false,
    accept = ".csv"
}: FileUploadProps) {
    const [file, setFile] = useState<File | null>(null)
    const [isDragOver, setIsDragOver] = useState(false)

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(true)
    }, [])

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(false)
    }, [])

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(false)

        const droppedFile = e.dataTransfer.files[0]
        if (droppedFile && droppedFile.name.endsWith('.csv')) {
            setFile(droppedFile)
        }
    }, [])

    const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0]
        if (selectedFile) {
            setFile(selectedFile)
        }
    }, [])

    const handleClear = useCallback(() => {
        setFile(null)
    }, [])

    const handleSubmit = useCallback(() => {
        if (file) {
            onFileSelect(file)
        }
    }, [file, onFileSelect])

    const formatFileSize = (bytes: number): string => {
        if (bytes < 1024) return `${bytes} B`
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    }

    return (
        <Card className="border-dashed border-2 hover:border-primary/50 transition-colors">
            <CardContent className="p-8">
                <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={cn(
                        "flex flex-col items-center justify-center gap-4 py-8 rounded-lg transition-colors",
                        isDragOver && "bg-primary/10"
                    )}
                >
                    {!file ? (
                        <>
                            <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center">
                                <Upload className="w-8 h-8 text-primary" />
                            </div>
                            <div className="text-center">
                                <h3 className="font-semibold text-lg mb-1">
                                    Upload Transaction Data
                                </h3>
                                <p className="text-muted-foreground text-sm">
                                    Drag and drop your CSV file here, or click to browse
                                </p>
                            </div>
                            <label className="cursor-pointer">
                                <input
                                    type="file"
                                    accept={accept}
                                    onChange={handleFileChange}
                                    className="hidden"
                                    disabled={isLoading}
                                />
                                <span className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all duration-200 border border-input bg-transparent shadow-sm hover:bg-accent hover:text-accent-foreground h-10 px-4 py-2">
                                    <FileText className="w-4 h-4" />
                                    Select CSV File
                                </span>
                            </label>
                        </>
                    ) : (
                        <>
                            <div className="w-16 h-16 rounded-full bg-emerald-500/20 flex items-center justify-center">
                                <FileText className="w-8 h-8 text-emerald-400" />
                            </div>
                            <div className="text-center">
                                <h3 className="font-semibold text-lg mb-1">{file.name}</h3>
                                <p className="text-muted-foreground text-sm">
                                    {formatFileSize(file.size)}
                                </p>
                            </div>
                            <div className="flex gap-3">
                                <Button
                                    variant="outline"
                                    onClick={handleClear}
                                    disabled={isLoading}
                                >
                                    <X className="w-4 h-4 mr-2" />
                                    Clear
                                </Button>
                                <Button onClick={handleSubmit} disabled={isLoading}>
                                    {isLoading ? (
                                        <>
                                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                            Analyzing...
                                        </>
                                    ) : (
                                        <>
                                            <Upload className="w-4 h-4 mr-2" />
                                            Analyze Transactions
                                        </>
                                    )}
                                </Button>
                            </div>
                        </>
                    )}
                </div>
            </CardContent>
        </Card>
    )
}
