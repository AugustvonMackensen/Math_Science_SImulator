import { MathfieldElement } from "mathlive";
import { useEffect, useRef } from "react";

// Load fonts/sounds from the CDN so the bundler doesn't need to serve them.
MathfieldElement.fontsDirectory = "https://unpkg.com/mathlive@0.105.3/dist/fonts";
MathfieldElement.soundsDirectory = null;

interface MathInputProps {
  value: string; // LaTeX
  onChange: (latex: string) => void;
  placeholder?: string;
}

// A MathLive math field. Emits its content as LaTeX, which the backend parses
// via SymPy (input_format: "latex").
export default function MathInput({ value, onChange, placeholder }: MathInputProps) {
  const ref = useRef<MathfieldElement>(null);

  useEffect(() => {
    const mf = ref.current;
    if (!mf) return;
    const handler = () => onChange(mf.value);
    mf.addEventListener("input", handler);
    return () => mf.removeEventListener("input", handler);
  }, [onChange]);

  // Keep the field in sync when the value is changed from outside.
  useEffect(() => {
    const mf = ref.current;
    if (mf && mf.value !== value) mf.value = value;
  }, [value]);

  return (
    <math-field ref={ref} placeholder={placeholder} style={{ width: "100%" }} />
  );
}
