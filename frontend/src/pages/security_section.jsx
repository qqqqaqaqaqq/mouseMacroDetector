import { useState, useRef, useEffect } from "react";
import SendData from './services/send_record';
import CircularSlider from "./slider_security";
import './styles/security_section.scss'

export default function Record() {
    const [isDragging, setIsDragging] = useState(false);
    const [record, setRecord] = useState([]);
    const areaRef = useRef(null);

    const [mouseleave, setMouseLeave] = useState(false)
    
    const last_ts = useRef(performance.now());
    
    const tolerance = 0.02;
    const MAX_QUEUE_SIZE = 100; // 반드시 75 이상

    const on_move = (e) => {
        if (isDragging == true) {
            const now_ts = performance.now();
            const delta = (now_ts - last_ts.current) / 1000;

            if (delta >= tolerance && areaRef.current) {
                // 1. 영역의 위치 정보 가져오기
                const rect = areaRef.current.getBoundingClientRect();

                // 2. 상대 좌표 계산 (정수 처리)
                const relX = Math.round(e.clientX - rect.left);
                const relY = Math.round(e.clientY - rect.top);

                const data = {
                    timestamp: new Date().toISOString(),
                    x: relX,
                    y: relY,
                    deltatime: Number(delta.toFixed(4))
                };

                last_ts.current = now_ts;

                setRecord(prev => [...prev, data]);
                console.log("기록 중(상대좌표):", data);
            }
        }
        else {
            setRecord([]);
        }   
    };

    const on_leave = () => {
        if (isDragging) {
            setIsDragging(false);
            setMouseLeave(true);
            setRecord([]); 
            console.log("영역을 벗어나 드래그가 취소되었습니다.");
        }
    };    
    
    const on_enter = () => {
        // 영역에 다시 들어오면 '취소 상태' 해제
        setMouseLeave(false);
    };

    useEffect(() => {
        if (record.length >= MAX_QUEUE_SIZE) {
            SendData(record);
            setRecord([]);
        }
    }, [record]);

    return (
        <div 
            className="security"
            ref={areaRef} 
            onMouseMove={on_move}
            onMouseLeave={on_leave}
            onMouseEnter={on_enter}        
        >
            <div className="security-area" >
                <CircularSlider 
                isDragging = {isDragging} 
                setIsDragging = {setIsDragging} 
                />
            
            </div>
            <p>현재 기록된 좌표 : {record.length}</p>
        </div>
    );
}