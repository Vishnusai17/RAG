
from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
OLD_PASSWORD = "neo4j"
NEW_PASSWORD = "secret123"

def change_password():
    print(f"Connecting to {URI} with old password...")
    try:
        # Connect with old credentials
        driver = GraphDatabase.driver(URI, auth=("neo4j", OLD_PASSWORD))
        
        # Change password query
        with driver.session(database="system") as session:
            print("Executing password change...")
            session.run(f"ALTER CURRENT USER SET PASSWORD FROM '{OLD_PASSWORD}' TO '{NEW_PASSWORD}'")
            print("✅ Password successfully changed to 'secret123'")
            
        driver.close()
        return True
    except Exception as e:
        print(f"❌ Failed to change password: {e}")
        return False

if __name__ == "__main__":
    change_password()
