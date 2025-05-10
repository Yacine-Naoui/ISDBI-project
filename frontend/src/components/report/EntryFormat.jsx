import { useState } from "react";
import { Send } from "lucide-react";

const EntryFormat = () => {
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const scenarioData = {
    title: "FAS 28 - Murabaha and Other Deferred Payment Sales",
    entries: [
      { type: "Dr", label: "Right of Use Asset (ROU)", amount: 489000 },
      { type: "Dr", label: "Deferred Ijarah Cost", amount: 111000 },
      { type: "Cr", label: "Ijarah Liability", amount: 600000 },
    ],
    amortization: {
      "Cost of ROU": 489000,
      "Terminal Value Difference": 2000,
      "Amortizable Amount": 487000,
    },
    explanation:
      "We deduct this 2,000 since the Lessee is expected to gain ownership, and this value will remain in the books after the lease ends.",
  };

  const handleSubmit = async () => {
    setLoading(true);
    setResponse(null);
    setError(null);

    try {
      const res = await fetch("/api/ai/process-scenario/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(scenarioData),
      });

      if (!res.ok) throw new Error("Server error");

      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setError("❌ Failed to process scenario. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full p-6 rounded-xl border border-gray-300 shadow-lg mb-6 flex flex-col gap-4 bg-white">
      <h2 className="text-xl font-semibold text-gray-800">
        {scenarioData.title}
      </h2>

      <div className="space-y-2">
        {scenarioData.entries.map((entry, index) => (
          <div
            key={index}
            className="flex justify-between bg-[#F9FAFB] border border-gray-200 p-3 rounded-md"
          >
            <span className="font-medium">
              {entry.type}. {entry.label}
            </span>
            <span className="font-mono text-right">
              ${entry.amount.toLocaleString()}
            </span>
          </div>
        ))}
      </div>

      <div className="border-t pt-4 text-sm text-gray-600">
        <p className="font-semibold mb-1">Amortizable Amount Calculation:</p>
        {Object.entries(scenarioData.amortization).map(([key, value], i) => (
          <div key={i} className="flex justify-between">
            <span>{key}</span>
            <span>${value.toLocaleString()}</span>
          </div>
        ))}
        <p className="mt-2 italic text-gray-500">{scenarioData.explanation}</p>
      </div>

      <button
        onClick={handleSubmit}
        disabled={loading}
        className="self-end mt-4 px-6 py-2 bg-ctaBlue text-white rounded-lg flex items-center gap-2 hover:bg-blue-700 transition-all duration-300 disabled:opacity-50"
      >
        <Send size={16} />
        {loading ? "Sending..." : "Submit Scenario"}
      </button>

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md text-sm text-red-800">
          {error}
        </div>
      )}

      {response && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-md text-sm text-green-800 overflow-x-auto">
          <pre className="whitespace-pre-wrap">{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default EntryFormat;