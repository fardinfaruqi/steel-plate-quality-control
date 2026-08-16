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
      `Example loaded from row ${selectedRow.index} (actual defect: ${selectedRow.defect}). Click 'Predict Defect' below to run the model.`,
    )
  }

  return (
    <div className="space-y-3">
      <p className="text-sm leading-relaxed text-steel-dim">
        Pick a defect type, then a row of that type from the held-out test set
        (rows the model never saw during training) to load into the form. The
        actual defect shown is only for reference; it is never used as model
        input. The model may not always predict the example correctly.
      </p>

      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block">
          <span className="field-label">Defect type</span>
          <div className="field">
            <select
              className="field-input field-select"
              value={selectedType}
              onChange={(e) => changeType(e.target.value)}
            >
              {counts.map(([defect, count]) => (
                <option key={defect} value={defect}>
                  {defect} ({count})
                </option>
              ))}
            </select>
          </div>
        </label>

        <label className="block">
          <span className="field-label">Choose an example (test-set row)</span>
          <div className="field">
            <select
              className="field-input field-select"
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
          </div>
        </label>
      </div>

      <button
        type="button"
        className="btn-secondary w-full cursor-pointer px-4 py-2.5"
        onClick={load}
        disabled={!selectedRow}
      >
        Load Example Values
      </button>

      {loaded && <Notice variant="success">{loaded}</Notice>}
    </div>
  )
}
