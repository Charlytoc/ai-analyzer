import { useState } from "react";
import { FileUploader } from "../Files/FileUploader";
import { Markdowner } from "../Markdowner/Markdowner";

export const SentenceBrief = () => {
  const [brief, setBrief] = useState("");
  const handleUploadSuccess = ({ brief }: { brief: string }) => {
    setBrief(brief);
  };

  return (
    <div>
      {brief && (
        <div className="mt-12 p-4 rounded-lg bg-white shadow-md">
          <h2 className="text-2xl font-bold ">Resumen de la sentencia</h2>
          <Markdowner markdown={brief} />
        </div>
      )}
      <div className="my-2 flex justify-center">
        <FileUploader onUploadSuccess={handleUploadSuccess} />
      </div>
    </div>
  );
};
