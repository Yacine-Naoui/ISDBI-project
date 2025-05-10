import { useState } from "react";
import ApplicableStandard from "../../components/report/ApplicableStandard";


const ReportTabsWrapper2 = ({ report }) => {
  const [activeTab, setActiveTab] = useState("results");

  const tabs = [
    { key: "results", label: "Results" },
    { key: "standards", label: "Standards" },
    { key: "history", label: "History" },
  ];

  return (
    <div className="border border-gray-300 h-full flex flex-col overflow-hidden">
      {/* Tab Selector */}
      <div className="sticky top-0 bg-white z-10 border-b border-gray-200 px-4 pt-3">
        <div className="flex gap-6 relative">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`pb-2 transition-colors duration-200 font-medium text-sm ${
                activeTab === tab.key
                  ? "text-blue-600"
                  : "text-gray-600 hover:text-blue-600"
              }`}
            >
              {tab.label}
              {activeTab === tab.key && (
                <div className="h-[2px] bg-blue-600 rounded-full mt-2"></div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Scrollable Content Area */}
      <div className="overflow-y-auto p-4 flex flex-col gap-5 custom-scrollbar">
        {activeTab === "results" && (
          <>
            <ApplicableStandard />
          </>
        )}

        {activeTab === "standards" && (
          <div className="text-sm text-gray-500">
            Standards content coming soon...
          </div>
        )}

        {activeTab === "history" && (
          <div className="text-sm text-gray-500">
            History content coming soon...
          </div>
        )}
      </div>
    </div>
  );
};

export default ReportTabsWrapper2;
