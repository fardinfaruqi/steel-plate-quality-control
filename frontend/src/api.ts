import type { Examples, Metadata } from './types'

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api'

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    throw new ApiError(`Request failed with status ${res.status}`, res.status)
  }
  return (await res.json()) as T
}

export function getMetadata(): Promise<Metadata> {
  return get<Metadata>('/metadata')
}

export function getExamples(): Promise<Examples> {
  return get<Examples>('/examples')
}

export async function predict(
  features: Record<string, number>,
): Promise<string> {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features }),
  })

  if (!res.ok) {
    let detail = `Prediction failed with status ${res.status}`
    try {
      const body = (await res.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // keep the fallback message
    }
    throw new ApiError(detail, res.status)
  }

  const body = (await res.json()) as { prediction: string }
  return body.prediction
}
