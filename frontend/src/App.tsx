import { useCallback, useEffect, useMemo, useState } from 'react'
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
      <div className="app-glow flex min-h-screen items-center justify-center px-6">
        <div className="panel max-w-md p-6 text-center">
          <p className="font-mono text-[0.7rem] uppercase tracking-[0.18em] text-molten">
            API unreachable
          </p>
          <p className="mt-2 text-sm text-steel">
            {metaError}. Make sure the API is running (uvicorn api.main:app
            --port 8000) and reload.
          </p>
        </div>
      </div>
    )
  }

  const featureCount = metadata?.feature_order.length ?? 27

  return (
    <div className="app-glow min-h-screen font-sans text-steel">
      <div className="mx-auto flex max-w-[1440px]">
        <aside className="sidebar-glow hidden w-72 shrink-0 border-r border-hairline lg:block">
          <div className="sticky top-0 p-6">
            {metadata && (
              <SidebarConsole
                modelInfo={metadata.model_info}
                noProbabilityNote={metadata.no_probability_note}
                featureCount={featureCount}
              />
            )}
          </div>
        </aside>

        <main className="min-w-0 flex-1 px-5 py-8 lg:px-10">
          <div className="lg:hidden">
            {metadata && (
              <SidebarConsole
                modelInfo={metadata.model_info}
                noProbabilityNote={metadata.no_probability_note}
                featureCount={featureCount}
              />
            )}
          </div>

          <Header />

          <div className="mt-4 space-y-3">
            <Expander title="About the model">
              <ul className="space-y-1.5 text-sm">
                {metadata?.about_model &&
                  Object.entries(metadata.about_model).map(([label, value]) => (
                    <li key={label}>
                      <span className="font-mono text-[0.72rem] uppercase tracking-[0.1em] text-steel-dim">
                        {label}:{' '}
                      </span>
                      <span className="text-hot-white">{value}</span>
                    </li>
                  ))}
              </ul>
            </Expander>

            <Expander title="Try an Example">
              {examples ? (
                <TryExample rows={examples} onLoad={handleLoadExample} />
              ) : (
                <p className="text-sm text-steel-dim">Loading examples…</p>
              )}
            </Expander>
          </div>

          <div className="my-6 border-t border-hairline" />

          <div className="mb-1 flex flex-col gap-1 sm:flex-row sm:items-baseline sm:gap-3">
            <h3 className="font-display text-[1.35rem] font-extrabold uppercase tracking-[0.02em] text-hot-white [font-stretch:125%]">
              Plate Measurements
            </h3>
            <span className="font-mono text-[0.72rem] uppercase tracking-[0.16em] text-molten">
              {featureCount} features · model order
            </span>
          </div>
          <p className="mb-4 text-sm leading-relaxed text-steel-dim">
            Enter the {featureCount} measured features. Feature names follow
            the trained model; values are reordered internally to the exact
            model order.
          </p>

          {metadata?.groups.map((group) => (
            <FeaturePanel
              key={group.slug}
              group={group}
              kinds={kinds}
              values={values}
              onValueChange={handleValueChange}
            />
          ))}

          <div className="my-6 border-t border-hairline" />

          <button
            type="button"
            className="btn-primary w-full cursor-pointer px-4 py-3"
            onClick={() => void handlePredict()}
            disabled={predicting || !metadata}
          >
            {predicting ? 'Predicting…' : 'Predict Defect'}
          </button>

          {result && (
            <ResultPanel
              prediction={result}
              cValue={metadata?.model_info.C ?? '10'}
              featureCount={featureCount}
            />
          )}

          {notice && <div className="mt-4">{notice.variant === 'error' ? <Notice variant="error">{notice.text}</Notice> : <Notice variant="success">{notice.text}</Notice>}</div>}

          <div className="my-6 border-t border-hairline" />

          <p className="text-xs leading-relaxed text-steel-dim">
            {metadata?.disclaimer}
          </p>
        </main>
      </div>
    </div>
  )
}
