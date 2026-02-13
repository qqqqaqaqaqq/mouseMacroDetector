export default async function SendData(data) {
  try {
    const res = await fetch(
      `${import.meta.env.VITE_POST_URL}/api/get_points`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }
    );

    const result = await res.json();
    const messageData = result.message;

    let raw_error = [];
    let threshold = 0.0;
    let raw_error_mean = 0.0;

    if (messageData && messageData.raw_error) {
      raw_error = messageData.raw_error;
    }
    
    if (messageData && messageData.threshold && messageData.threshold.length > 0) {
      threshold = messageData.threshold[0];
    }

    // --- 평균(Mean) 계산 로직 수정 ---
    if (raw_error.length > 0) {
      // reduce를 사용하여 배열의 합계를 구합니다.
      const sum = raw_error.reduce((acc, val) => acc + val, 0);
      raw_error_mean = sum / raw_error.length;
    } else {
      raw_error_mean = 0.0;
    }

    let human = false;

    if (raw_error_mean < threshold) {
      human = true
    }

    return {
      raw_error_mean: raw_error_mean,
      threshold:threshold,
      human:human
    };

  } catch (err) {
    console.error("SendData failed:", err);
    return false;
  }
}