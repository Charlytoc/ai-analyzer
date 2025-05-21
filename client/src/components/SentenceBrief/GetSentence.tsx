import { useEffect, useState } from "react";
import { getSentence } from "../../utils/api";
import { toast } from "react-hot-toast";
export const GetSentence = ({
  hash,
  onSuccess,
}: {
  hash: string;
  onSuccess: (sentence: string) => void;
}) => {
  const [sentence, setSentence] = useState("");

  const fetchSentence = async () => {
    const tid = toast.loading("Cargando...");
    try {
      const data = await getSentence(hash);
      console.log(data);
      onSuccess(data.brief);
      setSentence(data.brief);
      toast.success("Resumen obtenido correctamente", { id: tid });
    } catch (error) {
      console.error("Error al obtener resumen de la sentencia:", error);
      toast.error("Error al obtener resumen de la sentencia", { id: tid });
    }
  };
  useEffect(() => {
    fetchSentence();
  }, [hash]);

  return (
    <div>
      <h1>{sentence ? "Resumen de la sentencia obtenido" : "Cargando..."}</h1>
      <button
        className="button-edomex mt-4 px-4 py-2"
        onClick={() => fetchSentence()}
      >
        Reintentar
      </button>
    </div>
  );
};
