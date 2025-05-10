import FASCard from "../standards/FASCard";

const ApplicableStandard = () => {
  return (
    <div className="rounded-lg border border-gray-200 shadow-lg p-6 mb-6 bg-white">
      {/* Row 1 */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Applicable Standard</h3>
        <span className="bg-green-100 text-green-600 text-sm px-3 py-1 rounded-full">
          98% Match
        </span>
      </div>

      {/* Main Content (FAS Card) */}
      <div className="bg-gray-50 p-4 rounded-lg shadow-sm">
        <FASCard />
      </div>
    </div>
  );
};

export default ApplicableStandard;
