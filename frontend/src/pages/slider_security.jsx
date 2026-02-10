import React, { useRef, useState } from "react";
import { motion } from "framer-motion";
import "./styles/slider_security.scss";

export default function CircularSlider({isDragging, setIsDragging}) {

    return (
    <div className={`circular-container ${isDragging ? "active" : ""}`}>
        {/* 원을 잡았을 때 부모 영역에 스타일을 주고 싶다면 위처럼 클래스 제어가 가능합니다 */}
        
        <div className="canvas-area">
        <motion.div
            className="draggable-circle"
            drag
            dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
            dragElastic={0.5}
            
            // 드래그 시작 시
            onDragStart={() => {
            setIsDragging(true);
            }}
            
            // 드래그 종료 시
            onDragEnd={() => {
            setIsDragging(false);
            }}

            whileTap={{ scale: 0.9, cursor: "grabbing" }}
            whileHover={{ scale: 1.1 }}
        />
        </div>
    </div>
    );
}