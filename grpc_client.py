import grpc
import grpc_service_pb2
import grpc_service_pb2_grpc

# Sunucuya bağlan
channel = grpc.insecure_channel('localhost:50051')
stub = grpc_service_pb2_grpc.CommentAnalyzerStub(channel)

# Test için doğru bir topic_id gönder (Örnek: '007ACE74B050' dosyanızda var)
topic_id = "007ACE74B050"  # Örnek topic_id
response = stub.AnalyzeComment(grpc_service_pb2.AnalyzeRequest(topic_id=topic_id))
print(f"Sonuç: {response.conclusion}")
