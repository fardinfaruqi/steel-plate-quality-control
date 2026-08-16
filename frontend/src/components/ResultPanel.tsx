interface ResultPanelProps {
  prediction: string
  cValue: string
  featureCount: number
}

export function ResultPanel({ prediction, cValue, featureCount }: ResultPanelProps) {
  return (
    <div className="result-panel mt-6 px-6 py-6">
      <span className="mb-2 block font-mono text-[0.68rem] font-medium uppercase tracking-[0.18em] text-molten">
        Predicted defect
      </span>
      <h2 className="font-display text-[2.25rem] font-extrabold leading-tight text-hot-white [font-stretch:125%]">
        {prediction}
      </h2>
      <p className="mt-2 text-sm text-steel-dim">
        Decision from the tuned RBF SVM (C={cValue}) on {featureCount} plate
        measurements.
      </p>
    </div>
  )
}
