import { TrendingUp } from "lucide-react"; // Icon for going up trend
import { Line } from "react-chartjs-2"; // For Line Chart
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register necessary chart components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const BestDealChart = () => {
  // Dummy data for the line chart
  const chartData = {
    labels: ["0", "25", "50", "75", "100"], // Risk %
    datasets: [
      {
        label: "Shariah Compliance %",
        data: [10, 35, 60, 80, 90], // Dummy data for compliance %
        borderColor: "#22c55e", // Green color for the line
        backgroundColor: "rgba(34, 197, 94, 0.2)", // Light green fill under the line
        tension: 0.4, // Curved line
      },
    ],
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: false,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Risk %",
        },
      },
      y: {
        title: {
          display: true,
          text: "Shariah Compliance %",
        },
      },
    },
  };

  return (
    <div className="space-y-6 font-segoe">
      {/* Best Case Deal Component */}
      <div className=" border border-gray-200 rounded-lg shadow-lg p-6">
        {/* Title and Icon */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2 text-lg font-semibold text-gray-800">
            <TrendingUp className="" size={20} />
            Best Case Deal
          </div>
          <div className="flex items-center gap-2 text-sm ">
            <span className="w-2.5 h-2.5 rounded-full bg-green-500"></span>
            Current Case
          </div>
        </div>

        {/* Line Chart */}
        <Line data={chartData} options={chartOptions} />

        {/* Note Card */}
        <div className="mt-6 bg-green-50 text-green-800 p-4 rounded-lg ">
          <h3 className=" font-semibold">Optimization suggestion</h3>
          <p className="text-sm">Reducing profit margin from 15% to 12% would increase Shariah
          compliance by 8% with minimal risk impact.{" "}</p>
          
        </div>
      </div>
    </div>
  );
};

export default BestDealChart;
