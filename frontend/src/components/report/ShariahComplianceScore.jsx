import { Shield } from "lucide-react"; // Using the shield icon for the title

const ShariahComplianceScore = ({
  score = 85, // Default score (can be passed as a prop)
  structureScore = 90, // Default structure score
  contractsScore = 80, // Default contracts score
  processScore = 75, // Default process score
}) => {
  // Function to determine color based on score percentage
  const getColor = (percentage) => {
    if (percentage >= 80) return "text-green-400";
    if (percentage >= 60) return "text-yellow-400";
    if (percentage >= 40) return "text-orange-400";
    return "text-red-400";
  };

  // Average score calculation
  const averageScore = (structureScore + contractsScore + processScore) / 3;

  return (
    <div className="border border-gray-200 rounded-lg shadow-sm p-6 bg-white">
      <div className="flex items-center mb-6">
        <Shield className="" size={24} />
        <h3 className="ml-2 text-lg font-semibold text-gray-800">Shariah Compliance Score</h3>
      </div>

      {/* Circular Score Chart */}
      <div className="flex justify-center mb-6 relative">
        <div className="w-40 h-40 rounded-full border-8 border-gray-300 flex items-center justify-center bg-gray-200">
          {/* Disconnected Outer Border Segments (Top, Bottom, Left, Right) */}
          
          {/* Top part */}
          <div
            className={`absolute w-12 h-2 rounded-full ${getColor(score)} top-0 left-1/4`}
            style={{
              transform: "rotate(0deg)",
              position: "absolute",
              clipPath: "inset(0 0 50% 0)",
            }}
          />
          {/* Right part */}
          <div
            className={`absolute w-12 h-2 rounded-full ${getColor(score)} right-0 top-1/4`}
            style={{
              transform: "rotate(90deg)",
              position: "absolute",
              clipPath: "inset(0 0 50% 0)",
            }}
          />
          {/* Bottom part */}
          <div
            className={`absolute w-12 h-2 rounded-full ${getColor(score)} bottom-0 left-1/4`}
            style={{
              transform: "rotate(180deg)",
              position: "absolute",
              clipPath: "inset(0 0 50% 0)",
            }}
          />
          {/* Left part */}
          <div
            className={`absolute w-12 h-2 rounded-full ${getColor(score)} left-0 top-1/4`}
            style={{
              transform: "rotate(270deg)",
              position: "absolute",
              clipPath: "inset(0 0 50% 0)",
            }}
          />

          {/* Center score */}
          <div className={`absolute text-4xl font-bold text-gray-800 ${getColor(score)} `}>{score}%</div>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-100 p-4 rounded-lg text-center">
          <h4 className="text-sm font-semibold text-gray-700">Structure</h4>
          <div className={`text-2xl font-semibold ${getColor(structureScore)}`}>{structureScore}%</div>
        </div>
        <div className="bg-gray-100 p-4 rounded-lg text-center">
          <h4 className="text-sm font-semibold text-gray-700">Contracts</h4>
          <div className={`text-2xl font-semibold ${getColor(contractsScore)}`}>{contractsScore}%</div>
        </div>
        <div className="bg-gray-100 p-4 rounded-lg text-center">
          <h4 className="text-sm font-semibold text-gray-700">Process</h4>
          <div className={`text-2xl font-semibold ${getColor(processScore)}`}>{processScore}%</div>
        </div>
      </div>
    </div>
  );
};

export default ShariahComplianceScore;
