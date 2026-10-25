import { Fragment, useCallback, useEffect, useMemo, useState } from 'react'
import { getExamples, getMetadata, predict } from './api'
import type { ExampleRow, FeatureKind, Metadata } from './types'
import { Header } from './components/Header'
import { SidebarConsole } from './components/SidebarConsole'
import { Expander } from './components/Expander'
import { TryExample } from './components/TryExample'
import { FeaturePanel } from './components/FeaturePanel'
import { ResultPanel } from './components/ResultPanel'
import { Notice } from './components/Notice'

type NoticeState = { variant: 'success' | 'error'; text: string } | null

const LEAD =
  'Predict the defect type on a steel plate from 27 measured features.'

export default function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null)
  const [examples, setExamples] = useState<ExampleRow[] | null>(null)
  const [metaError, setMetaError] = useState<string | null>(null)

  const [values, setValues] = useState<Record<string, number>>({})
  const [result, setResult] = useState<string | null>(null)
  const [notice, setNotice] = useState<NoticeState>(null)
  const [predicting, setPredicting] = useState(false)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const [meta, ex] = await Promise.all([getMetadata(), getExamples()])
        if (cancelled) return
        setMetadata(meta)
        setExamples(ex.rows)
      } catch (err) {
        if (cancelled) return
        setMetaError(err instanceof Error ? err.message : 'Failed to load the application.')
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!metadata) return
    const zeros: Record<string, number> = {}
    for (const feature of metadata.feature_order) zeros[feature] = 0
    setValues(zeros)
  }, [metadata])

  const kinds = useMemo(() => {
    const map = new Map<string, FeatureKind>()
    if (!metadata) return map
    for (const f of metadata.binary_features) map.set(f, 'binary')
    for (const f of metadata.integer_features) {
      if (!map.has(f)) map.set(f, 'integer')
    }
    for (const f of metadata.feature_order) {
      if (!map.has(f)) map.set(f, 'decimal')
    }
    return map
  }, [metadata])

  const handleValueChange = useCallback((feature: string, value: number) => {
    setValues((prev) => ({ ...prev, [feature]: value }))
  }, [])

  const handleLoadExample = useCallback(
    (row: ExampleRow) => {
      if (!metadata) return
      const next: Record<string, number> = {}
      for (let i = 0; i < metadata.feature_order.length; i += 1) {
        next[metadata.feature_order[i]] = row.values[i] ?? 0
      }
      setValues(next)
      setResult(null)
      setNotice(null)
    },
    [metadata],
  )

  const handlePredict = useCallback(async () => {
    setPredicting(true)
    setResult(null)
    setNotice(null)
    try {
      const prediction = await predict(values)
      setResult(prediction)
      setNotice({
        variant: 'success',
        text: 'Prediction completed using the tuned SVM model.',
      })
    } catch (err) {
      setNotice({
        variant: 'error',
        text: err instanceof Error ? err.message : 'Prediction failed.',
      })
    } finally {
      setPredicting(false)
    }
  }, [values])

  if (metaError) {
    return (
      <div className="app">
        <Header />
        <main id="main" className="workspace container">
          <div className="state">
            <h2 className="state__title">The model did not load</h2>
            <p className="state__body">
              {metaError}. Make sure the API is running (uvicorn
              api.main:app --port 8000) and reload the page.
            </p>
          </div>
        </main>
      </div>
    )
  }

  if (!metadata) {
    return (
      <div className="app">
        <Header />
        <main id="main" className="workspace container">
          <p className="lead">{LEAD}</p>
          <p className="visually-hidden" role="status">
            Loading the model metadata and the held-out rows.
          </p>
          <div className="work" aria-busy="true">
            <div className="column">
              <div className="skeleton skeleton--strip" />
              <div className="skeleton skeleton--console" />
            </div>
            <div className="rail">
              <div className="skeleton skeleton--rail" />
            </div>
          </div>
        </main>
      </div>
    )
  }

  const featureCount = metadata.feature_order.length

  return (
    <div className="app">
      <a className="skip" href="#main">
        Skip to content
      </a>

      <Header />

      <main id="main" className="workspace container">
        <p className="lead">{LEAD}</p>

        <div className="work">
          <div className="column">
            {examples && (
              <TryExample rows={examples} onLoad={handleLoadExample} />
            )}

            <section className="console" aria-labelledby="measurements">
              <div className="console__head">
                <div className="card__head">
                  <h2 className="card__title" id="measurements">
                    Plate measurements
                  </h2>
                  <span className="card__count">{featureCount} features</span>
                </div>
                <p className="card__note">
                  Enter the measurements, or load a held-out row to fill them.
                  Values are sent in the model’s own order.
                </p>
              </div>
              {metadata.groups.map((group) => (
                <FeaturePanel
                  key={group.slug}
                  group={group}
                  kinds={kinds}
                  values={values}
                  onValueChange={handleValueChange}
                />
              ))}
            </section>
          </div>

          <aside className="rail">
            <div className="rail__sticky">
              <button
                type="button"
                className="btn btn--primary btn--block"
                onClick={() => void handlePredict()}
                disabled={predicting || !metadata}
                aria-busy={predicting}
              >
                {predicting ? 'Predicting…' : 'Predict defect'}
              </button>

              {result && (
                <ResultPanel
                  prediction={result}
                  cValue={metadata.model_info.C ?? '10'}
                  featureCount={featureCount}
                />
              )}

              {notice?.variant === 'error' && (
                <Notice variant="error">{notice.text}</Notice>
              )}
            </div>

            <SidebarConsole
              modelInfo={metadata.model_info}
              noProbabilityNote={metadata.no_probability_note}
              featureCount={featureCount}
            />
          </aside>
        </div>

        <section className="about">
          <Expander title="About the model">
            <dl className="record__list">
              {Object.entries(metadata.about_model).map(([label, value]) => (
                <Fragment key={label}>
                  <dt>{label}</dt>
                  <dd>{value}</dd>
                </Fragment>
              ))}
            </dl>
          </Expander>
        </section>
      </main>

      <footer className="foot">
        <div className="container">
          <p className="disclaimer">{metadata.disclaimer}</p>
        </div>
      </footer>
    </div>
  )
}