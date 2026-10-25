import { Fragment } from 'react'

interface SidebarConsoleProps {
  modelInfo: Record<string, string>
  noProbabilityNote: string
  featureCount: number
}

export function SidebarConsole({
  modelInfo,
  noProbabilityNote,
  featureCount,
}: SidebarConsoleProps) {
  const accuracy = modelInfo['Test Accuracy']
  const rows = Object.entries(modelInfo).filter(
    ([label]) => label !== 'Test Accuracy',
  )

  return (
    <section className="card" aria-labelledby="model-record">
      <div className="card__head">
        <h2 className="card__title" id="model-record">
          Model
        </h2>
        <span className="card__count">{featureCount} features</span>
      </div>

      {accuracy && (
        <dl className="record__headline">
          <dt>Test accuracy</dt>
          <dd>{accuracy}</dd>
        </dl>
      )}

      <dl className="record__list">
        {rows.map(([label, value]) => (
          <Fragment key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </Fragment>
        ))}
      </dl>
      <p className="record__note">{noProbabilityNote}</p>
    </section>
  )
}