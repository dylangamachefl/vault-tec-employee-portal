"""SQLAlchemy ORM models for the auth layer: Role and User."""

from __future__ import annotations

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.auth.database import Base


class Role(Base):
    """A role definition with its allowed Qdrant access_level values."""

    __tablename__ = "roles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    # Human-readable slug, e.g. "admin"
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    # Display name, e.g. "IT Administrator"
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    # List of AccessLevelV2 strings the role may retrieve from Qdrant
    # e.g. ["General Employee", "HR Restricted"]
    allowed_access_levels: Mapped[list] = mapped_column(JSON, nullable=False)

    users: Mapped[list[User]] = relationship("User", back_populates="role")

    def __repr__(self) -> str:
        return f"<Role {self.name!r} levels={self.allowed_access_levels}>"


class User(Base):
    """A demo portal user linked to a Role."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # "u1" … "u4"
    username: Mapped[str] = mapped_column(String, nullable=False)
    role_id: Mapped[str] = mapped_column(String, ForeignKey("roles.id"), nullable=False)

    role: Mapped[Role] = relationship("Role", back_populates="users")

    def __repr__(self) -> str:
        return f"<User {self.id!r} username={self.username!r} role={self.role_id!r}>"
