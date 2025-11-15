from sqlalchemy import Column, Integer, String, Date, Float
from db import Base

class TblDataset(Base):
    __tablename__ = "tblDataset"
    id = Column("id", Integer, primary_key=True, autoincrement=True)
    name = Column("name", String(255))
    path = Column("path", String(255))
    type = Column("type", String(255))
    tblModelid = Column("tblModelid", Integer)

class TblDatasetDetail(Base):
    __tablename__ = "tblDatasetDetail"
    id = Column("id", Integer, primary_key=True, autoincrement=True)
    tblSampleid = Column("tblSampleid", Integer, nullable=False)
    tblDatasetid = Column("tblDatasetid", Integer, nullable=False)

class TblSample(Base):
    __tablename__ = "tblSample"
    id = Column("id", Integer, primary_key=True, autoincrement=True)
    nameImg = Column("nameImg", String(255))
    path = Column("path", String(255))
    createDate = Column("createDate", Date)
    type = Column("type", String(255))

class TblLabel(Base):
    __tablename__ = "tblLabel"
    id = Column("id", Integer, primary_key=True, autoincrement=True)
    xTop = Column("xTop", Float)
    yTop = Column("yTop", Float)
    xBot = Column("xBot", Float)
    yBot = Column("yBot", Float)
    label = Column("label", String(255))
    tblSampleid = Column("tblSampleid", Integer, nullable=False)
