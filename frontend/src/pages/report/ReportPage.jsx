import Layout from "../../components/layout/Layout";

import Welcome from "../../components/home/Welcome";
import UseCase from "../../components/home/UseCase";


import ReportTabsWrapper from "./ReportTabWrapper";
import ChatBot from "../../components/home/ChatBot";

const ReportPage = () => {


  return (
    <Layout>
      {/* Main Content Area */}
      <main className="flex justify-between">
        <div className="flex flex-col gap-5 basis-[43%]">
          {/* <Welcome /> */}
          <ChatBot />
        </div>
        <div className="flex flex-col basis-[55%] max-h-[calc(100vh-4rem)] overflow-hidden">
          <ReportTabsWrapper />
          
        </div>
      </main>
    </Layout>
  );
};

export default ReportPage;
