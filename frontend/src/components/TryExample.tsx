import { useMemo, useState } from 'react'
import type { ExampleRow } from '../types'
import { Notice } from './Notice'

interface TryExampleProps {
  rows: ExampleRow[]
  onLoad: (row: ExampleRow) => void
}

export function TryExample({ rows, onLoad }: TryExampleProps) {
  const counts = useMemo(() => {
    const map = new Map<string, number>()
    for (const row of rows) {
      map.set(row.defect, (map.get(row.defect) ?? 0) + 1)
    }
    return [...map.entries()].sort((a, b) => b[1] - a[1])
  }, [rows])

  const [selectedType, setSelectedType] = useState(counts[0]?.[0] ?? '')
  const rowsForType = useMemo(
    () => rows.filter((r) => r.defect === selectedType),
    [rows, selectedType],
  )
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [loaded, setLoaded] = useState<string | null>(null)

  const selectedRow = rowsForType[selectedIndex] ?? rowsForType[0]

  function changeType(defect: string) {
    setSelectedType(defect)
    setSelectedIndex(0)
    setLoaded(null)
  }

  function load() {
    if (!selectedRow) return
    onLoad(selectedRow)
    setLoaded(
      `Row ${selectedRow.index} loaded. Actual defect ${selectedRow.defect}. The model never saw this row.`,
    )
  }

  return (
    <section className="card" aria-labelledby="example-title">
      <div className="card__head">
        <h2 className="card__title" id="example-title">
          Load a held-out row
        </h2>
        <span className="card__count">
          {rows.length} {rows.length === 1 ? 'row' : 'rows'}
        </span>
      </div>

      {rows.length === 0 ? (
        <p className="card__note">
          The API returned no test rows, so there is nothing to load. Restart
          the service (uvicorn api.main:app --port 8000) and reload.
        </p>
      ) : (
        <>
          <p className="card__note">
            Pick a defect type, then a held-out row the model never saw during
            training. The actual defect is for reference only; it is never
            model input.
          </p>

          <div className="example__row">
            <label className="field">
              <span className="field__label">Defect type</span>
              <span className="field__wrap field__wrap--select">
                <select
                  className="field__input field__input--select"
                  value={selectedType}
                  onChange={(e) => changeType(e.target.value)}
                >
                  {counts.map(([defect, count]) => (
                    <option key={defect} value={defect}>
                      {defect} ({count})
                    </option>
                  ))}
                </select>
              </span>
            </label>

            <label className="field">
              <span className="field__label">Test-set row</span>
              <span className="field__wrap field__wrap--select">
                <select
                  className="field__input field__input--select"
                  value={selectedRow ? selectedRow.index : ''}
                  onChange={(e) => setSelectedIndex(Number(e.target.value))}
                  disabled={rowsForType.length === 0}
                >
                  {rowsForType.map((row) => (
                    <option key={row.index} value={row.index}>
                      Row {row.index}
                    </option>
                  ))}
                </select>
              </span>
            </label>

            <button
              type="button"
              className="btn"
              onClick={load}
              disabled={!selectedRow}
            >
              Load row
            </button>
          </div>

          {loaded && (
            <div className="example__status">
              <Notice variant="status">{loaded}</Notice>
            </div>
          )}
        </>
      )}
    </section>
  )
}