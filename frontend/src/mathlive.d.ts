import type { MathfieldElement } from "mathlive";
import type React from "react";

// Allow <math-field> as a JSX intrinsic element (MathLive web component).
declare global {
  namespace JSX {
    interface IntrinsicElements {
      "math-field": React.DetailedHTMLProps<
        React.HTMLAttributes<MathfieldElement>,
        MathfieldElement
      > & { placeholder?: string };
    }
  }
}
