import axios from "axios";
import { useState } from "react";

export default function Comparator() {
  const [targetText, setTargetText] = useState("");
  const [compares, setCompares] = useState([""]);
  const [func, setFunc] = useState("cosine");
  const [results, setResults] = useState<{ text: string; score: number }[]>([]);
  const [loading, setLoading] = useState(false);

  const handleCompareChange = (idx: number, value: string) => {
    const newCompares = [...compares];
    newCompares[idx] = value;
    setCompares(newCompares);
  };

  const addCompareField = () => setCompares([...compares, ""]);
  const removeCompareField = (idx: number) =>
    setCompares(compares.filter((_, i) => i !== idx));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await axios.post(
        "http://localhost:8005/api/compare-embeddings",
        {
          target_text: targetText,
          compares: compares.filter((t) => t.trim() !== ""),
          function: func,
        }
      );
      setResults(res.data.results);
    } catch (err) {
      console.error(err);
      alert("❌ " + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  // Helper para truncar
  const truncate = (s: string, n = 30) =>
    s.length > n ? s.slice(0, n) + "…" : s;

  return (
    <div className="max-w-xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Comparator</h1>
      <form onSubmit={handleSubmit}>
        {/* Target Text */}
        <label className="block mb-2">Target text (ターゲットテキスト):</label>
        <textarea
          className="w-full border rounded p-2 mb-4"
          rows={2}
          value={targetText}
          onChange={(e) => setTargetText(e.target.value)}
          required
        />

        {/* Function Select */}
        <label className="block mb-2">Function (関数):</label>
        <select
          className="w-full border rounded p-2 mb-4"
          value={func}
          onChange={(e) => setFunc(e.target.value)}
        >
          <option value="cosine">Cosine Similarity (コサイン類似度)</option>
          <option value="euclidean">Euclidean Distance (距離)</option>
        </select>

        {/* Compare Fields */}
        <label className="block mb-2">Texts to compare (比較テキスト):</label>
        {compares.map((text, idx) => (
          <div key={idx} className="flex items-center mb-2">
            <input
              className="flex-1 border rounded p-2 mr-2"
              type="text"
              value={text}
              onChange={(e) => handleCompareChange(idx, e.target.value)}
              placeholder="Escribe aquí…"
              required
            />
            {compares.length > 1 && (
              <button
                type="button"
                onClick={() => removeCompareField(idx)}
                className="text-red-500"
              >
                ✕
              </button>
            )}
          </div>
        ))}
        <button
          type="button"
          onClick={addCompareField}
          className="mb-4 text-blue-600"
        >
          ➕ Añadir texto
        </button>

        {/* Submit */}
        <div>
          <button
            type="submit"
            disabled={loading}
            className="bg-green-500 text-white px-4 py-2 rounded"
          >
            {loading ? "Enviando…" : "Comparar"}
          </button>
        </div>
      </form>

      {/* Results Inline */}
      {results.length > 0 && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-2">Resultados:</h2>
          {results.map((r, i) => (
            <details
              key={i}
              className="border rounded mb-2 overflow-hidden"
            >
              <summary className="flex justify-between bg-gray-100 p-2 cursor-pointer">
                <span>{truncate(r.text)}</span>
                <span className="font-mono font-bold">{r.score.toFixed(4)}</span>
              </summary>
              <div className="p-2 bg-white">
                {r.text}
              </div>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
