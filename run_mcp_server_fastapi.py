#!/usr/bin/env python3
"""
多模式MCP随机森林服务器启动脚本

支持两种通信模式：
1. stdio模式 - 标准输入输出，适用于传统MCP客户端
2. SSE模式 - Server-Sent Events，适用于CherryStudio等Web客户端

使用方法：
  python run_mcp_server.py --mode stdio    # stdio模式
  python run_mcp_server.py --mode sse      # SSE模式，默认端口8080
  python run_mcp_server.py --mode sse --host 0.0.0.0 --port 9000  # 自定义SSE配置
"""

import sys
import os
import argparse
import asyncio
import uvicorn
from pathlib import Path
import glob
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 导入MCP相关模块
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server import Server
# 导入我们的MCP服务器
from src.mcp_svm_tool.mcp_server import mcp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from starlette.requests import Request


def create_fastapi_app(mcp_server: Server, *, debug: bool = False) -> FastAPI:
    """为SSE模式创建FastAPI应用."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request):
        """处理SSE连接."""
        # 添加初始化延迟，确保服务器完全准备好
        await asyncio.sleep(0.5)

        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    # 创建 FastAPI 实例
    app = FastAPI(
        title="MCP Random Forest Server",
        description="多模式MCP随机森林服务器 - 支持MCP协议和RESTful API",
        version="1.0.0",
        debug=debug
    )

    # 添加CORS中间件（跨域支持）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源，生产环境建议指定具体域名
        allow_credentials=True,
        allow_methods=["*"],  # 允许所有HTTP方法
        allow_headers=["*"],  # 允许所有请求头
    )

    # 挂载静态文件服务 - 为trained_models目录提供静态文件访问
    trained_models_path = project_root / "trained_models"
    if trained_models_path.exists():
        app.mount("/static", StaticFiles(directory=str(trained_models_path)), name="static")

    # 添加路由
    app.add_route("/sse", handle_sse)
    app.mount("/messages/", sse.handle_post_message)
    
    # 使用装饰器添加额外的 API 路由
    @app.get("/")
    async def root():
        """根端点，显示服务器信息"""
        return {
            "message": "MCP XGBoost Server with CORS enabled",
            "server": "FastAPI",
            "cors_enabled": True,
            "endpoints": {
                "sse": "/sse",
                "hello": "/hello",
                "health": "/health",
                "reports": "/reports",
                "report": "/report/{model_id}/{report_type}",
                "static": "/static",
                "download_model": "/download/{model_id}",
                "download_file": "/download/file/{file_path}",
                "archives": "/archives",
                "reports_browser": "/reports/browser"
            }
        }
    
    @app.get("/hello")
    async def say_hello(name: str = "World"):
        """Hello端点，支持跨域访问"""
        return {
            "message": f"Hello, {name}!",
            "server": "FastAPI with CORS",
            "cors_enabled": True
        }
    
    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return {
            "status": "healthy",
            "server": "MCP Random Forest",
            "cors_enabled": True
        }
    
    @app.get("/reports")
    async def list_reports():
        """列出所有可用的HTML报告"""
        try:
            reports = []
            trained_models_dir = project_root / "trained_models"
            
            if not trained_models_dir.exists():
                return {"reports": [], "message": "No trained models directory found"}
            
            # 遍历所有模型目录
            for model_dir in trained_models_dir.iterdir():
                if model_dir.is_dir() and model_dir.name != "__pycache__":
                    model_id = model_dir.name
                    model_reports = {"model_id": model_id, "reports": []}
                    
                    # 查找训练报告
                    training_report = model_dir / "training_report.html"
                    if training_report.exists():
                        stat = training_report.stat()
                        model_reports["reports"].append({
                            "type": "training",
                            "filename": "training_report.html",
                            "url": f"/report/{model_id}/training",
                            "static_url": f"/static/{model_id}/training_report.html",
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                    
                    # 查找预测报告
                    predictions_dir = model_dir / "predictions"
                    if predictions_dir.exists():
                        for pred_dir in predictions_dir.iterdir():
                            if pred_dir.is_dir():
                                pred_report = pred_dir / "prediction_report.html"
                                if pred_report.exists():
                                    stat = pred_report.stat()
                                    model_reports["reports"].append({
                                        "type": "prediction",
                                        "prediction_id": pred_dir.name,
                                        "filename": "prediction_report.html",
                                        "url": f"/report/{model_id}/prediction/{pred_dir.name}",
                                        "static_url": f"/static/{model_id}/predictions/{pred_dir.name}/prediction_report.html",
                                        "size": stat.st_size,
                                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                                    })
                    
                    # 查找特征分析报告
                    feature_analysis_dir = model_dir / "feature_analysis" / "local"
                    if feature_analysis_dir.exists():
                        for analysis_dir in feature_analysis_dir.iterdir():
                            if analysis_dir.is_dir():
                                for html_file in analysis_dir.glob("*.html"):
                                    stat = html_file.stat()
                                    model_reports["reports"].append({
                                        "type": "feature_analysis",
                                        "analysis_id": analysis_dir.name,
                                        "filename": html_file.name,
                                        "url": f"/report/{model_id}/feature_analysis/{analysis_dir.name}/{html_file.name}",
                                        "static_url": f"/static/{model_id}/feature_analysis/local/{analysis_dir.name}/{html_file.name}",
                                        "size": stat.st_size,
                                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                                    })
                    
                    # 只添加有报告的模型
                    if model_reports["reports"]:
                        reports.append(model_reports)
            
            return {
                "total_models": len(reports),
                "total_reports": sum(len(model["reports"]) for model in reports),
                "reports": reports
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing reports: {str(e)}")
    
    @app.get("/report/{model_id}/training")
    async def get_training_report(model_id: str):
        """获取指定模型的训练报告"""
        try:
            report_path = project_root / "trained_models" / model_id / "training_report.html"
            if not report_path.exists():
                raise HTTPException(status_code=404, detail=f"Training report not found for model {model_id}")
            
            return FileResponse(
                path=str(report_path),
                media_type="text/html",
                filename=f"training_report_{model_id}.html"
            )
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Error serving training report: {str(e)}")
    
    @app.get("/report/{model_id}/prediction/{prediction_id}")
    async def get_prediction_report(model_id: str, prediction_id: str):
        """获取指定模型和预测ID的预测报告"""
        try:
            report_path = project_root / "trained_models" / model_id / "predictions" / prediction_id / "prediction_report.html"
            if not report_path.exists():
                raise HTTPException(status_code=404, detail=f"Prediction report not found for model {model_id}, prediction {prediction_id}")
            
            return FileResponse(
                path=str(report_path),
                media_type="text/html",
                filename=f"prediction_report_{model_id}_{prediction_id}.html"
            )
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Error serving prediction report: {str(e)}")
    
    @app.get("/report/{model_id}/feature_analysis/{analysis_id}/{filename}")
    async def get_feature_analysis_report(model_id: str, analysis_id: str, filename: str):
        """获取指定模型的特征分析报告"""
        try:
            report_path = project_root / "trained_models" / model_id / "feature_analysis" / "local" / analysis_id / filename
            if not report_path.exists():
                raise HTTPException(status_code=404, detail=f"Feature analysis report not found: {filename}")
            
            return FileResponse(
                path=str(report_path),
                media_type="text/html",
                filename=f"feature_analysis_{model_id}_{analysis_id}_{filename}"
            )
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=f"Error serving feature analysis report: {str(e)}")
    
    @app.get("/download/{model_id}")
    async def download_model_archive(model_id: str):
        """下载模型的完整压缩包（通过模型ID查找最新压缩包）"""
        try:
            # 查找模型目录
            model_dir = trained_models_path / model_id
            if not model_dir.exists():
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            # 查找压缩文件
            archives_dir = trained_models_path / "archives"
            if not archives_dir.exists():
                raise HTTPException(status_code=404, detail="No archives directory found")
            
            # 查找最新的压缩文件（按模型ID匹配）
            archive_files = list(archives_dir.glob(f"{model_id}_*.zip"))
            if not archive_files:
                raise HTTPException(status_code=404, detail=f"No archive found for model {model_id}")
            
            # 选择最新的压缩文件
            latest_archive = max(archive_files, key=lambda x: x.stat().st_mtime)
            
            # 返回文件响应
            return FileResponse(
                path=str(latest_archive),
                filename=latest_archive.name,
                media_type='application/zip',
                headers={
                    "Content-Disposition": f"attachment; filename={latest_archive.name}",
                    "Content-Description": f"Complete model package for {model_id}"
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error downloading model archive: {str(e)}")

    @app.get("/download/file/{file_path:path}")
    async def download_zip_file(file_path: str):
        """通用下载接口 - 直接通过文件路径下载ZIP文件"""
        try:
            # 构建完整的文件路径
            full_path = project_root / file_path
            
            # 安全检查：确保文件在项目根目录内
            try:
                full_path.resolve().relative_to(project_root.resolve())
            except ValueError:
                raise HTTPException(status_code=403, detail="Access denied: file outside project directory")
            
            # 检查文件是否存在
            if not full_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            
            # 检查是否为文件（不是目录）
            if not full_path.is_file():
                raise HTTPException(status_code=400, detail=f"Path is not a file: {file_path}")
            
            # 检查文件扩展名（可选：只允许特定类型）
            allowed_extensions = {'.zip', '.tar', '.gz', '.tar.gz', '.rar', '.7z'}
            if full_path.suffix.lower() not in allowed_extensions:
                raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed: {', '.join(allowed_extensions)}")
            
            # 确定MIME类型
            mime_types = {
                '.zip': 'application/zip',
                '.tar': 'application/x-tar',
                '.gz': 'application/gzip',
                '.tar.gz': 'application/gzip',
                '.rar': 'application/x-rar-compressed',
                '.7z': 'application/x-7z-compressed'
            }
            media_type = mime_types.get(full_path.suffix.lower(), 'application/octet-stream')
            
            # 返回文件响应
            return FileResponse(
                path=str(full_path),
                filename=full_path.name,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename={full_path.name}",
                    "Content-Description": f"Archive file: {full_path.name}",
                    "X-File-Path": file_path  # 添加原始路径信息
                }
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")
    
    @app.get("/archives")
    async def list_archives():
        """列出所有可用的模型压缩包"""
        try:
            archives = []
            archives_dir = trained_models_path / "archives"
            
            if not archives_dir.exists():
                return {"archives": [], "message": "No archives directory found"}
            
            # 遍历所有压缩文件
            for archive_file in archives_dir.glob("*.zip"):
                if archive_file.is_file():
                    stat = archive_file.stat()
                    # 从文件名提取模型ID（假设格式为 model_id_timestamp.zip）
                    filename_parts = archive_file.stem.split('_')
                    if len(filename_parts) >= 2:
                        model_id = '_'.join(filename_parts[:-1])  # 除了最后一个时间戳部分
                        timestamp = filename_parts[-1]
                    else:
                        model_id = archive_file.stem
                        timestamp = "unknown"
                    
                    archives.append({
                        "model_id": model_id,
                        "filename": archive_file.name,
                        "timestamp": timestamp,
                        "download_url": f"/download/{model_id}",
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            # 按创建时间排序（最新的在前）
            archives.sort(key=lambda x: x["created"], reverse=True)
            
            return {
                "total_archives": len(archives),
                "archives": archives
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing archives: {str(e)}")

    @app.get("/reports/browser", response_class=HTMLResponse)
    async def reports_browser():
        """提供一个简单的HTML界面来浏览所有报告"""
        html_content = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MCP Random Forest - 报告浏览器</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 30px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                }
                .model-section {
                    margin-bottom: 30px;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    overflow: hidden;
                }
                .model-header {
                    background-color: #f8f9fa;
                    padding: 15px;
                    font-weight: bold;
                    color: #495057;
                    border-bottom: 1px solid #ddd;
                }
                .reports-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 15px;
                    padding: 20px;
                }
                .report-card {
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    padding: 15px;
                    background-color: #fff;
                    transition: box-shadow 0.2s;
                }
                .report-card:hover {
                    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                }
                .report-type {
                    font-weight: bold;
                    color: #007bff;
                    margin-bottom: 8px;
                }
                .report-details {
                    font-size: 0.9em;
                    color: #6c757d;
                    margin-bottom: 10px;
                }
                .report-link {
                    display: inline-block;
                    background-color: #007bff;
                    color: white;
                    padding: 8px 15px;
                    text-decoration: none;
                    border-radius: 4px;
                    transition: background-color 0.2s;
                }
                .report-link:hover {
                    background-color: #0056b3;
                }
                .loading {
                    text-align: center;
                    padding: 50px;
                    color: #6c757d;
                }
                .error {
                    color: #dc3545;
                    text-align: center;
                    padding: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>MCP Random Forest - 报告浏览器</h1>
                <div id="content">
                    <div class="loading">正在加载报告列表...</div>
                </div>
            </div>

            <script>
                async function loadReports() {
                    try {
                        const response = await fetch('/reports');
                        const data = await response.json();
                        
                        const contentDiv = document.getElementById('content');
                        
                        if (data.reports.length === 0) {
                            contentDiv.innerHTML = '<div class="error">暂无可用报告</div>';
                            return;
                        }
                        
                        let html = `<p style="text-align: center; color: #6c757d;">
                            共找到 ${data.total_models} 个模型，${data.total_reports} 个报告
                        </p>`;
                        
                        data.reports.forEach(model => {
                            html += `
                                <div class="model-section">
                                    <div class="model-header">
                                        模型 ID: ${model.model_id}
                                        <span style="float: right;">${model.reports.length} 个报告</span>
                                    </div>
                                    <div class="reports-grid">
                            `;
                            
                            model.reports.forEach(report => {
                                const reportTypeName = {
                                    'training': '训练报告',
                                    'prediction': '预测报告',
                                    'feature_analysis': '特征分析报告'
                                }[report.type] || report.type;
                                
                                html += `
                                    <div class="report-card">
                                        <div class="report-type">${reportTypeName}</div>
                                        <div class="report-details">
                                            文件: ${report.filename}<br>
                                            大小: ${(report.size / 1024).toFixed(1)} KB<br>
                                            修改时间: ${new Date(report.modified).toLocaleString('zh-CN')}
                                        </div>
                                        <a href="${report.url}" class="report-link" target="_blank">查看报告</a>
                                    </div>
                                `;
                            });
                            
                            html += `
                                    </div>
                                </div>
                            `;
                        });
                        
                        contentDiv.innerHTML = html;
                        
                    } catch (error) {
                        document.getElementById('content').innerHTML = 
                            `<div class="error">加载报告失败: ${error.message}</div>`;
                    }
                }
                
                // 页面加载完成后获取报告列表
                document.addEventListener('DOMContentLoaded', loadReports);
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    return app

async def run_stdio_mode():
    """运行stdio模式."""
    print("[启动] MCP神经网络服务器 (stdio模式)")
    print("[配置] 模型将保存到: ./trained_model")
    print("[提示] 此模式适用于传统MCP客户端")
    print("="*50)
    
    # 直接运行FastMCP服务器
    await mcp.run()

def run_sse_mode(host: str = "0.0.0.0", port: int = 8080, debug: bool = True):
    """运行SSE模式."""
    print("[启动] MCP神经网络服务器 (SSE模式)")
    print(f"[网络] 服务器地址: http://{host}:{port}")
    print(f"[端点] SSE端点: http://{host}:{port}/sse")
    print("[配置] 模型将保存到: ./trained_model")
    print("[提示] 此模式适用于CherryStudio等Web客户端")
    print("="*50)
    
    # 获取底层MCP服务器并创建Starlette应用
    mcp_server = mcp._mcp_server
    starlette_app = create_fastapi_app(mcp_server, debug=debug)
    
    # 使用uvicorn运行服务器
    uvicorn.run(
        starlette_app, 
        host=host, 
        port=port, 
        log_level="debug" if debug else "info"
    )

def main():
    """主函数，解析命令行参数并启动相应模式."""
    parser = argparse.ArgumentParser(
        description="MCP神经网络服务器 - 支持多种通信模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --mode stdio                    # stdio模式（传统MCP客户端）
  %(prog)s --mode sse                      # SSE模式（Web客户端，默认8080端口）
  %(prog)s --mode sse --port 9000          # SSE模式，自定义端口
  %(prog)s --mode sse --host 127.0.0.1     # SSE模式，绑定本地地址

支持的工具函数:
  - train_neural_network / train_neural_network_from_content
  - train_classification_model / train_classification_from_content
  - predict_from_file / predict_from_content
  - predict_from_values
  - list_models
  - get_model_info
  - delete_model
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["stdio", "sse"],
        default="stdio",
        help="通信模式：stdio（标准输入输出）或 sse（Server-Sent Events）"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="SSE模式的绑定地址（默认: 0.0.0.0）"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="SSE模式的端口号（默认: 8080）"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    print(f"[模式] 通信模式: {args.mode.upper()}")
    if args.mode == "sse":
        print(f"[网络] 网络配置: {args.host}:{args.port}")
    if args.debug:
        print("[调试] 调试模式: 启用")
    
    try:
        if args.mode == "stdio":
            # stdio模式
            asyncio.run(run_stdio_mode())
        elif args.mode == "sse":
            # SSE模式（同步函数，因为uvicorn.run是同步的）
            run_sse_mode(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n[停止] 服务器已停止")
    except Exception as e:
        print(f"[错误] 服务器启动失败: {e}")
        import traceback
        if args.debug:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 