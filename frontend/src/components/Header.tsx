export function Header() {
  return (
    <header>
      <p className="mb-2 font-mono text-[0.72rem] font-medium uppercase tracking-[0.22em] text-molten">
        Plate inspection station · defect classifier
      </p>
      <h1 className="font-display text-[clamp(2rem,4.2vw,3.25rem)] font-extrabold leading-none tracking-tight text-hot-white [font-stretch:125%]">
        Steel Plate Defect Predictor
      </h1>
      <p className="mt-3 max-w-[62ch] text-base leading-relaxed text-steel-dim">
        This application uses a trained Support Vector Machine (SVM) model to
        predict the type of defect detected on a steel plate based on 27
        measured features.
      </p>
      <div className="heat-strip mt-7 mb-1" aria-hidden="true" />
    </header>
  )
}
