# agent3/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from django.core.files.base import ContentFile
from .models import PDFReport
from .serializers import PDFReportSerializer
from users.permissions import IsVerifiedBankUser
from agent3 import enhance_standard_with_graph
from CreateEmbedding import vector_store


class PDFProcessingView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [IsVerifiedBankUser]

    def post(self, request):
        # Create PDFReport instance with uploaded file
        serializer = PDFReportSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        pdf_report = serializer.save(user=request.user)

        try:
            # Process predefined standards
            standards_to_enhance = ["FAS 4", "FAS 10", "FAS 32"]
            results = {}

            for standard_id in standards_to_enhance:
                try:
                    results[standard_id] = enhance_standard_with_graph(
                        standard_id=standard_id, vector_store=vector_store
                    )
                except Exception as e:
                    results[standard_id] = {"status": "failed", "error": str(e)}

            # Generate PDF report content
            report_content = self._generate_report_content(results)
            pdf_buffer = self._create_pdf_buffer(report_content)

            # Save output PDF to model
            pdf_report.output_pdf.save(
                f"enhanced_{pdf_report.input_pdf.name}",
                ContentFile(pdf_buffer.getvalue()),
            )
            pdf_report.save()

            return Response(
                PDFReportSerializer(pdf_report).data, status=status.HTTP_201_CREATED
            )

        except Exception as e:
            pdf_report.delete()
            return Response(
                {"error": f"PDF processing failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _generate_report_content(self, results):
        """Generate formatted text report from processing results"""
        report_lines = [
            "# AAOIFI Standards Enhancement Report",
            "## Processing Results\n",
        ]

        for standard_id, result in results.items():
            status = " Success" if result.get("status") == "completed" else " Failed"
            report_lines.append(f"### {standard_id}")
            report_lines.append(f"**Status:** {status}")

            if result.get("final_report"):
                report_lines.append(f"**Summary:** {result['final_report'][:300]}...")
            elif result.get("error"):
                report_lines.append(f"**Error:** {result['error']}")

            report_lines.append("\n---\n")

        return "\n".join(report_lines)

    def _create_pdf_buffer(self, content):
        """Create PDF buffer from text content using ReportLab"""
        from reportlab.pdfgen import canvas
        from io import BytesIO

        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        text = c.beginText(40, 800)

        # Set up font styles
        text.setFont("Helvetica", 12)

        # Process content lines
        for line in content.split("\n"):
            if line.startswith("# "):
                text.setFont("Helvetica-Bold", 16)
                text.textLine(line[2:])
            elif line.startswith("## "):
                text.setFont("Helvetica-Bold", 14)
                text.textLine(line[3:])
            elif line.startswith("### "):
                text.setFont("Helvetica-Bold", 12)
                text.textLine(line[4:])
            else:
                text.setFont("Helvetica", 12)
                text.textLine(line)

        c.drawText(text)
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer
