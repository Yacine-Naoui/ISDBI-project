import Layout from "../../components/layout/Layout";
import PromptForm from "../../components/report/PromptForm";
import ReportTabsWrapper2 from "./ReportTabWrapper2";

const ReportPage2 = () => {
  return (
    <Layout>
      {/* Main Content Area */}
      <main className="flex justify-between">
        {/* Left Section: Prompt Form */}
        <div className="flex flex-col gap-5 basis-[43%]">
          <PromptForm />
        </div>

        {/* Right Section: Report Tabs */}
        <div className="flex flex-col basis-[55%] max-h-[calc(100vh-4rem)] overflow-hidden">
          <ReportTabsWrapper2 />
        </div>
      </main>
    </Layout>
  );
};

export default ReportPage2;
