import { AlertTriangle } from "lucide-react";

const RiskAnalysis = () => {
  // Dummy data for the risk metrics
  const riskData = [
    { title: "Shariah Compliance", percentage: 85 },
    { title: "Credit Risk", percentage: 70 },
    { title: "Market Risk", percentage: 60 },
    { title: "Operational Risk", percentage: 50 },
  ];

  // Function to determine the color of the bar based on percentage
  const getBarColor = (percentage) => {
    if (percentage >= 80) return "bg-green-400";
    if (percentage >= 60) return "bg-yellow-400";
    if (percentage >= 40) return "bg-orange-400";
    return "bg-red-400";
  };

  return (
    <div className="border border-gray-200 rounded-lg shadow-sm p-6 bg-white">
      <div className="flex items-center mb-6">
        <AlertTriangle className="" size={24} />
        <h3 className="ml-2 text-lg font-semibold text-gray-800">Risk Analysis</h3>
      </div>

      {/* Risk Metrics */}
      {riskData.map((risk, index) => (
        <div key={index} className="mb-4">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-gray-700">{risk.title}</span>
            <span className="text-sm font-medium text-gray-700">{risk.percentage}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full mt-2">
            <div
              className={`h-2 rounded-full ${getBarColor(risk.percentage)} `}
              style={{ width: `${risk.percentage}%` }}
            ></div>
          </div>
        </div>
      ))}

      {/* Recommendation Note */}
      <div className="bg-[#EFF6FF] text-[#1E40AF] p-4 rounded-lg mt-6">
        <h4 className="font-semibold">Recommendation</h4>
        <p className="text-sm mt-2">
          Consider additional documentation to mitigate operational risk factors.
        </p>
      </div>
    </div>
  );
};

export default RiskAnalysis;
