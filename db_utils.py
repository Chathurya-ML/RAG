from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# --------------------
# Database Setup
# --------------------
DB_URL = "sqlite:///rag.db"   # Change this to use PostgreSQL or MySQL later
engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


# --------------------
# Models
# --------------------
class ApplicationLog(Base):
    __tablename__ = "application_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, index=True)
    user_query = Column(Text)
    gpt_response = Column(Text)
    model = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class DocumentStore(Base):
    __tablename__ = "document_store"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)


# --------------------
# CRUD Operations
# --------------------
def init_db():
    Base.metadata.create_all(bind=engine)


def insert_application_log(session_id, user_query, gpt_response, model):
    with SessionLocal() as session:
        log = ApplicationLog(
            session_id=session_id,
            user_query=user_query,
            gpt_response=gpt_response,
            model=model,
        )
        session.add(log)
        session.commit()


def get_chat_history(session_id):
    with SessionLocal() as session:
        logs = (
            session.query(ApplicationLog)
            .filter_by(session_id=session_id)
            .order_by(ApplicationLog.created_at)
            .all()
        )
        messages = []
        for log in logs:
            messages.extend([
                {"role": "human", "content": log.user_query},
                {"role": "ai", "content": log.gpt_response},
            ])
        return messages


def insert_document_record(filename):
    with SessionLocal() as session:
        doc = DocumentStore(filename=filename)
        session.add(doc)
        session.commit()
        return doc.id


def delete_document_record(file_id):
    with SessionLocal() as session:
        doc = session.query(DocumentStore).filter_by(id=file_id).first()
        if doc:
            session.delete(doc)
            session.commit()
            return True
        return False


def get_all_documents():
    with SessionLocal() as session:
        docs = session.query(DocumentStore).order_by(DocumentStore.upload_timestamp.desc()).all()
        return [
            {"id": d.id, "filename": d.filename, "upload_timestamp": d.upload_timestamp}
            for d in docs
        ]


# --------------------
# Initialize Tables
# --------------------
init_db()
