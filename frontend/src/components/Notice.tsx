import type { ReactNode } from 'react'

interface NoticeProps {
  variant: 'success' | 'error'
  children: ReactNode
}

export function Notice({ variant, children }: NoticeProps) {
  const tone =
    variant === 'success'
      ? 'border-molten-line bg-mill-panel text-steel'
      : 'border-cooling bg-[rgb(27,36,48/0.6)] text-hot-white'

  return (
    <div
      role={variant === 'error' ? 'alert' : 'status'}
      className={`border ${tone} rounded-[10px] px-4 py-3 text-sm`}
    >
      {children}
    </div>
  )
}
