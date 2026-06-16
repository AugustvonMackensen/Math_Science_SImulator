import katex from "katex";
import { useMemo } from "react";

interface TeXProps {
  expr: string;
  display?: boolean;
}

// Render a LaTeX string with KaTeX. Errors are shown inline rather than thrown.
export default function TeX({ expr, display = false }: TeXProps) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(expr, {
        displayMode: display,
        throwOnError: false,
        errorColor: "#c0392b",
      });
    } catch {
      return `<span style="color:#c0392b">invalid TeX</span>`;
    }
  }, [expr, display]);

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
