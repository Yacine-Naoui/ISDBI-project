import ChatBot from "../../components/home/ChatBot";
import Layout from "../../components/layout/Layout";
import ReportTabsWrapper3 from "./ReportTabWrapper3";

const ReportPage3 = () => {
  return (
    <Layout>
      {/* Main Content Area */}
      <main className="flex justify-between">
        {/* Left Section: Prompt Form */}
        <div className="flex flex-col gap-5 basis-[43%]">
          <ChatBot />
        </div>

        {/* Right Section: Report Tabs */}
        <div className="flex flex-col basis-[55%] max-h-[calc(100vh-4rem)] overflow-hidden">
          <ReportTabsWrapper3 />
        </div>
      </main>
    </Layout>
  );
};

export default ReportPage3;
