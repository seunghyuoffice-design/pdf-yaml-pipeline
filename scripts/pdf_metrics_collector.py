#!/usr/bin/env python3
"""
PDF Parsing Metrics Reporter

Collects and reports metrics on PDF parsing success/failure rates,
parser effectiveness, and error patterns.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

try:
    import redis
except ImportError:
    redis = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class PDFMetricsCollector:
    """Collects and reports PDF parsing metrics."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None

        if redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Connected to Redis for metrics collection")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None

    def collect_file_metrics(self) -> Dict[str, Any]:
        """Collect metrics from Redis file queues."""
        if not self.redis_client:
            return {}

        try:
            # Get queue statistics
            queue_length = self.redis_client.llen("file:queue")
            processing_count = self.redis_client.llen("file:queue:processing")
            done_count = self.redis_client.scard("file:done")
            failed_count = self.redis_client.scard("file:failed")

            total_processed = done_count + failed_count
            success_rate = (done_count / total_processed * 100) if total_processed > 0 else 0

            return {
                "queue_length": queue_length,
                "processing_count": processing_count,
                "completed_count": done_count,
                "failed_count": failed_count,
                "total_processed": total_processed,
                "success_rate": round(success_rate, 2),
                "failure_rate": round(100 - success_rate, 2) if total_processed > 0 else 0
            }
        except Exception as e:
            logger.error(f"Failed to collect file metrics: {e}")
            return {}

    def collect_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns from failed files."""
        if not self.redis_client:
            return {}

        try:
            failed_files = self.redis_client.smembers("file:failed")
            error_patterns = {
                "total_failed": len(failed_files),
                "error_types": {},
                "parser_failures": {},
                "file_extensions": {}
            }

            for file_key in failed_files:
                # Get file extension
                ext = Path(file_key).suffix.lower()
                error_patterns["file_extensions"][ext] = error_patterns["file_extensions"].get(ext, 0) + 1

                # Try to get error details (if stored)
                error_key = f"error:{file_key}"
                error_data = self.redis_client.get(error_key)
                if error_data:
                    try:
                        error_info = json.loads(error_data)
                        error_type = error_info.get("error_type", "unknown")
                        parser_used = error_info.get("parser", "unknown")

                        error_patterns["error_types"][error_type] = error_patterns["error_types"].get(error_type, 0) + 1
                        error_patterns["parser_failures"][parser_used] = error_patterns["parser_failures"].get(parser_used, 0) + 1
                    except json.JSONDecodeError:
                        pass

            return error_patterns
        except Exception as e:
            logger.error(f"Failed to collect error patterns: {e}")
            return {}

    def collect_parser_effectiveness(self) -> Dict[str, Any]:
        """Collect parser effectiveness metrics."""
        if not self.redis_client:
            return {}

        try:
            # Get parser usage stats (if stored)
            parser_stats = {}
            parser_keys = self.redis_client.keys("parser:*:stats")

            for key in parser_keys:
                parser_name = key.split(":")[1]
                stats_data = self.redis_client.get(key)
                if stats_data:
                    try:
                        stats = json.loads(stats_data)
                        parser_stats[parser_name] = stats
                    except json.JSONDecodeError:
                        pass

            return parser_stats
        except Exception as e:
            logger.error(f"Failed to collect parser effectiveness: {e}")
            return {}

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "period": "current",
            "file_metrics": self.collect_file_metrics(),
            "error_patterns": self.collect_error_patterns(),
            "parser_effectiveness": self.collect_parser_effectiveness(),
            "recommendations": []
        }

        # Generate recommendations based on metrics
        file_metrics = report["file_metrics"]
        error_patterns = report["error_patterns"]

        if file_metrics.get("failure_rate", 0) > 50:
            report["recommendations"].append(
                f"High failure rate ({file_metrics['failure_rate']}%) - investigate primary parser issues"
            )

        if error_patterns.get("error_types"):
            most_common_error = max(error_patterns["error_types"], key=error_patterns["error_types"].get)
            if most_common_error == "corrupted":
                report["recommendations"].append("High corrupted PDF rate - implement pre-validation")
            elif most_common_error == "memory":
                report["recommendations"].append("Memory issues detected - reduce worker memory or batch sizes")

        return report

    def save_report(self, output_path: str = "pdf_metrics_report.json") -> None:
        """Save metrics report to file."""
        report = self.generate_report()

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Metrics report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def print_report(self) -> None:
        """Print metrics report to console."""
        report = self.generate_report()

        print("=== PDF Parsing Metrics Report ===")
        print(f"Generated: {report['timestamp']}")
        print()

        file_metrics = report["file_metrics"]
        if file_metrics:
            print("📊 File Processing Metrics:")
            print(f"  Queue Length: {file_metrics.get('queue_length', 0)}")
            print(f"  Currently Processing: {file_metrics.get('processing_count', 0)}")
            print(f"  Completed: {file_metrics.get('completed_count', 0)}")
            print(f"  Failed: {file_metrics.get('failed_count', 0)}")
            print(f"  Success Rate: {file_metrics.get('success_rate', 0)}%")
            print()

        error_patterns = report["error_patterns"]
        if error_patterns.get("error_types"):
            print("❌ Error Patterns:")
            for error_type, count in error_patterns["error_types"].items():
                print(f"  {error_type}: {count}")
            print()

        if error_patterns.get("parser_failures"):
            print("🔧 Parser Failures:")
            for parser, count in error_patterns["parser_failures"].items():
                print(f"  {parser}: {count}")
            print()

        parser_effectiveness = report["parser_effectiveness"]
        if parser_effectiveness:
            print("⚡ Parser Effectiveness:")
            for parser, stats in parser_effectiveness.items():
                success_rate = stats.get('success_rate', 0)
                usage_count = stats.get('attempts', 0)
                print(f"  {parser}: {usage_count} attempts, {success_rate}% success")
            print()

        if report["recommendations"]:
            print("💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"  • {rec}")
            print()

def main():
    """Main entry point for metrics collection."""
    import argparse

    parser = argparse.ArgumentParser(description="PDF Parsing Metrics Collector")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--output", help="Output file path (JSON)")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")

    args = parser.parse_args()

    collector = PDFMetricsCollector(args.redis_host, args.redis_port)

    if not args.quiet:
        collector.print_report()

    if args.output:
        collector.save_report(args.output)

if __name__ == "__main__":
    main()