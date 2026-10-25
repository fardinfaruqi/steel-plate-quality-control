interface ResultPanelProps {
  prediction: string
  cValue: string
  featureCount: number
}

export function ResultPanel({ prediction, cValue }: ResultPanelProps) {
  return (
    <dl className="reading">
      <dt className="reading__term">Predicted defect</dt>
      <dd className="reading__value">{prediction}</dd>
      <dd className="reading__note">
        Tuned RBF SVM, C={cValue}, over the entered measurements.
      </dd>
    </dl>
  )
}